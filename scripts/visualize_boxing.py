import sys
import os
import glob
import re
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from systems.sft_system import SFTSystem
from data.boxing import BoxingDataset
from data.skeleton import EDGES_BODY25

def render_boxing_comparison_mp4(gt_pose, pred_pose, text_name, output_mp4, fps=60):
    """
    Render 3D skeleton comparison video:
    GT (Blue) vs Prediction (Orange/Red)
    gt_pose: (T, 25, 3)
    pred_pose: (T, 25, 3)
    """
    seq_len = gt_pose.shape[0]
    
    fig = plt.figure(figsize=(12, 6))
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")
    
    all_pts = np.concatenate([gt_pose, pred_pose], axis=0)
    min_v, max_v = np.min(all_pts, axis=(0, 1)), np.max(all_pts, axis=(0, 1))
    pad = 0.2
    x_lim = [min_v[0] - pad, max_v[0] + pad]
    y_lim = [min_v[1] - pad, max_v[1] + pad]
    z_lim = [min_v[2] - pad, max_v[2] + pad]
    
    frames = []
    for t in range(seq_len):
        ax_gt.clear()
        ax_pred.clear()
        
        for ax, pose, title, color in [(ax_gt, gt_pose[t], "Ground Truth (GT)", "royalblue"),
                                     (ax_pred, pred_pose[t], f"Prediction ({text_name})", "darkorange")]:
            ax.set_xlim(x_lim); ax.set_ylim(z_lim); ax.set_zlim([-y_lim[1], -y_lim[0]])
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
            ax.view_init(elev=15, azim=-70)
            
            xs, ys, zs = pose[:, 0], pose[:, 2], -pose[:, 1]
            ax.scatter(xs, ys, zs, c=color, s=25)
            for v1, v2 in EDGES_BODY25:
                if v1 < len(pose) and v2 < len(pose):
                    ax.plot([xs[v1], xs[v2]], [ys[v1], ys[v2]], [zs[v1], zs[v2]], color=color, linewidth=2)
                    
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8").reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())
        
    plt.close(fig)
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    imageio.mimsave(output_mp4, frames, fps=fps)
    print(f"✅ 渲染完成: {output_mp4}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="outputs/sft_boxing_2026-07-28_01-49")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epoch", type=int, default=-1, help="Epoch to load (-1 for latest or best)")
    parser.add_argument("--latest", action="store_true", help="Force load latest checkpoint by epoch number")
    parser.add_argument("--fps", type=int, default=60, help="Rendering video FPS (default: 60)")
    args = parser.parse_args()
    
    exp_dir = os.path.abspath(args.exp_dir)
    cfg_path = os.path.join(exp_dir, ".hydra", "config.yaml")
    config = OmegaConf.load(cfg_path)
    
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        print(f"❌ 找不到 checkpoint 在 {ckpt_dir}")
        return

    def get_epoch_num(path):
        name = os.path.basename(path)
        m = re.search(r"epoch_1based=(\d+)", name) or re.search(r"epoch=(\d+)", name)
        return int(m.group(1)) if m else 0

    sorted_ckpts = sorted(ckpts, key=get_epoch_num)
    
    if args.latest or args.epoch == -1:
        # Load the latest checkpoint by highest epoch number
        target_ckpt = sorted_ckpts[-1]
    else:
        target_ckpt = None
        for c in sorted_ckpts:
            if get_epoch_num(c) == args.epoch:
                target_ckpt = c
                break
        if target_ckpt is None:
            target_ckpt = sorted_ckpts[-1]
        
    print(f"📦 載入 Checkpoint: {target_ckpt}")
    device = torch.device(args.device)
    
    system = SFTSystem(config=config, target_dim=25 * 3)
    ckpt = torch.load(target_ckpt, map_location="cpu")
    system.load_state_dict(ckpt["state_dict"], strict=False)
    system.to(device)
    system.eval()
    
    # 載入 Boxing 測試集 (split=2)
    test_ds = BoxingDataset(
        data_dir=config.dataset.data_dir,
        input_n=config.dataset.input_n,
        output_n=config.dataset.output_n,
        skip_rate=config.dataset.skip_rate,
        split=2,
        random_face=False
    )
    
    loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    ep_num = get_epoch_num(target_ckpt)
    out_dir = os.path.join(exp_dir, "visualize_samples", f"epoch_{ep_num:03d}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 開始為 Boxing 測試集 (Checkpoint Epoch {ep_num}) 產生 {args.num_samples} 筆視覺化樣本...")
    
    sample_cnt = 0
    for batch_idx, batch in enumerate(loader):
        if sample_cnt >= args.num_samples:
            break
            
        motion_name = batch.get("motion_name", ["boxing_action"])[0]
        pose_gt = batch["pose"].float().to(device) # (1, T, 75)
        mask = batch["mask"].float().to(device) # (1, T, 75)
        timepoints = batch.get("timepoints")
        if timepoints is not None:
            timepoints = timepoints.float().to(device)
        else:
            timepoints = torch.arange(pose_gt.shape[1], device=device).unsqueeze(0).float()
            
        gt_feed = {"pose": pose_gt, "mask": mask, "timepoints": timepoints}
        
        with torch.no_grad():
            tok_emb, tok_mask = system.text_encoder([motion_name])
            text_emb = (tok_emb, tok_mask)
            pred_out, _, _, _ = system.model.evaluate(gt_feed, n_samples=1, text_embedding=text_emb)
            
        # pred_out shape: (1, 1, 75, T) -> (T, 25, 3)
        pred_pose = pred_out[0, 0].cpu().numpy().transpose(1, 0).reshape(-1, 25, 3)
        gt_pose_3d = pose_gt[0].cpu().numpy().reshape(-1, 25, 3)
        
        safe_name = re.sub(r'[^\w\-_\. ]', '_', motion_name)
        mp4_path = os.path.join(out_dir, f"sample_{sample_cnt+1}_{safe_name}.mp4")
        
        render_boxing_comparison_mp4(gt_pose_3d, pred_pose, motion_name, mp4_path, fps=args.fps)
        sample_cnt += 1

    print(f"🎉 全部 {sample_cnt} 筆視覺化樣本處理完畢！結果歸檔於: {out_dir}")

if __name__ == "__main__":
    main()
