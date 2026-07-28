import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
#將/home/allen/Diffusion/ReasonMotion_SFT_GRPO加至系統路徑
import sys
sys.path.append("/home/allen/Diffusion/ReasonMotion_SFT_GRPO_Trajectory")
from utils.finefs import FineFS                                       # 統一 dataset
from utils.fs_reward_utils.fs_reward_model import HumanPosePerception, CoachScoringModel

'''
python train.py --epochs 200 --batch_size 64
'''



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train FS Reward Model (GOE Predictor)")
    parser.add_argument("--data_dir", type=str, default="/home/allen/datasets/FineFS_5s/3_final", help="Dataset root")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_goe", help="Where to save models")
    parser.add_argument("--layout", type=str, default="SMPL_24", help="Skeleton layout (SMPL or SMPL_24)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Loaders
    print("Initializing Datasets...")
    train_dataset = FineFS(args.data_dir, input_n=30, output_n=40, split=0, mode="rotation",
                           move_global=True, random_face=True, reward_mode=True)
    val_dataset   = FineFS(args.data_dir, input_n=30, output_n=40, split=1, mode="rotation",
                           move_global=True, random_face=False, reward_mode=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Model
    print("Initializing Model...")
    hpp_backbone = HumanPosePerception(
        num_class=1024, in_channel=6, residual=True, dropout=0.5, 
        t_kernel_size=9, layout=args.layout, strategy='spatial', 
        hop_size=3, num_att_graph=4, hpp_way='HPP', pretrain=True
    )
    model = CoachScoringModel(hpp_backbone).to(device)
    
    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 4. Training Loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            inputs = batch['pose'].to(device)
            targets = batch['judge_score'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['pose'].to(device)
                targets = batch['judge_score'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Validation Metrics
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {mae:.4f}")
        
        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path = os.path.join(args.checkpoint_dir, "best_model_goe.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'config': vars(args)
            }, path)
            print(f"-> Saved best model to {path}")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
