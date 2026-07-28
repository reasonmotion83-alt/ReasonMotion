import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
import pickle
import argparse
from pathlib import Path

# Add project root to sys.path
PRJ_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(PRJ_ROOT))

from model import ModelMain
from utils.text_encoder import TextEncoder
from utils.config_util import load_config
from utils.finefs import FineFS
from torch.utils.data import DataLoader

# SMPL_24 Edges
EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18),
    (18, 20), (20, 22), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
]

def plot_skeleton(ax, joint_coords, title="Skeleton"):
    # joint_coords: (24, 3)
    # Flip Y for visualization: in this dataset Y-positive is down, so we plot -Y as height.
    y_vis = -joint_coords[:, 1]
    ax.scatter(joint_coords[:, 0], y_vis, c='red', s=20)
    
    bone_lengths = []
    for (v1, v2) in EDGES:
        p1 = joint_coords[v1]
        p2 = joint_coords[v2]
        ax.plot([p1[0], p2[0]], [-p1[1], -p2[1]], color='blue', alpha=0.5)
        
        dist = np.linalg.norm(p1 - p2)
        bone_lengths.append(dist)
        
        # Label distance at the flipped mid-point
        mid = (p1 + p2) / 2
        ax.text(mid[0], -mid[1], f"{dist:.3f}", fontsize=7, color='green')

    y_min_vis, y_max_vis = np.min(y_vis), np.max(y_vis)
    height = y_max_vis - y_min_vis
    ax.set_title(f"{title}\nHeight: {height:.3f}")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(PRJ_ROOT / "configs/train_grpo.yaml"))
    parser.add_argument("--ckpt", type=str, default="/home/allen/Diffusion/DePOSit_Skating_Predictor_BaseModel/runs/rotation_30_40_0711-1541/checkpoints/model_ep30.pth")
    parser.add_argument("--save", type=str, default="scale_comparison.png")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Config & Model
    config = load_config(args.config)
    model = ModelMain(config, device=device, target_dim=24 * 3).to(device)
    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    text_encoder = TextEncoder(device=str(device))

    # 2. Load Data (GT)
    # Speed up: use only a few files
    print("📂 Loading Data...")
    dataset = FineFS(
        data_dir=config['data']['data_dir'],
        input_n=config['data']['input_n'],
        output_n=config['data']['output_n'],
        skip_rate=1,
        split=1, # Validation
        mode=config['data']['mode'],
        max_len=config['data']['max_len'],
        data_ratio=0.01
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    
    motion_name = batch['motion_name'][0]
    print(f"Sample Motion: {motion_name}")

    # 3. Inference (Following visualize_infer_basemodel.py logic)
    print("✨ Performing Inference...")
    with torch.no_grad():
        tok_emb, tok_mask = text_encoder([motion_name])
        tok_emb, tok_mask = tok_emb.to(device), tok_mask.to(device)
        
        # Batch preparation - Keep standard (B, T, K) as expected by model.process_data
        pose = batch["pose"].to(device).float() # (1, 70, 72)
        tp = batch["timepoints"].to(device).float() # (1, 70)
        mask = batch["mask"].to(device).float() # (1, 70, 72)
        
        # Mask everything for pure generation like visualize_infer_basemodel.py does with --all_mask
        mask[:, :, :] = 0.0
        
        feed = {"pose": pose, "mask": mask, "timepoints": tp}
        
        # model.evaluate expects (feed, n_samples, text_embedding=(emb, mask))
        # returns (samples, gt, obs_mask, tp)
        # samples shape: (B, n, K, L) -> (1, 1, 72, 70)
        samples_out = model.evaluate(feed, 1, text_embedding=(tok_emb, tok_mask))[0]
        
        # Reshape generated
        # (1, 1, 72, 70) -> (72, 70) -> (70, 72) -> (70, 24, 3)
        gen_pose = samples_out[0, 0].permute(1, 0).cpu().numpy().reshape(70, 24, 3)
        
        # Reshape GT
        # batch['pose'] is (1, 70, 72)
        gt_pose = batch['pose'][0].numpy().reshape(70, 24, 3)

    # 4. Visualization
    print("🖼 Creating Visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    # Pick mid frame
    frame_idx = 60
    
    plot_skeleton(axes[0], gt_pose[frame_idx], title=f"GT Skeleton ({motion_name})")
    plot_skeleton(axes[1], gen_pose[frame_idx], title=f"Generated Skeleton ({motion_name})")

    plt.tight_layout()
    plt.savefig(args.save)
    print(f"✅ Comparison saved to {args.save}")

if __name__ == "__main__":
    main()


"""
python utils/fs_reward_utils/verify_data.py --save scale_check_upright.png
"""