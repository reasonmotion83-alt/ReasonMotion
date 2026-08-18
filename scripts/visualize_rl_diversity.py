import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
ReasonMotion RL Diversity and Parameter Sweep Visualizer
This script performs an offline diversity parameter sweep for RL/SFT models,
testing different values of sampling_std and active exploration step counts (num_discrete_steps/window_size),
and visually compares the diversity of variant branches using skeleton grids laid out in 3D space.

Usage examples:
1. Sweep different sampling_std values:
   python scripts/visualize_rl_diversity.py --exp_dir outputs/rl_finefs_2026-07-15_00-03 --sweep_type std --sampling_std 0.02 0.05 0.1 0.2 0.5 --num_discrete_steps 5

2. Sweep different numbers of active steps:
   python scripts/visualize_rl_diversity.py --exp_dir outputs/rl_finefs_2026-07-15_00-03 --sweep_type steps --sampling_std 0.1 --num_discrete_steps 5 10 20 50

3. Test fully independent DDPM noise (without delayed sub-trajectory):
   python scripts/visualize_rl_diversity.py --exp_dir outputs/rl_finefs_2026-07-15_00-03 --sweep_type std --sampling_std 0.02 0.05 0.1 0.2 --no_delayed
"""
import glob
import re
import argparse
import pickle
import numpy as np
import torch
import imageio
import random
from tqdm import tqdm
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Import project-internal modules
from systems.sft_system import SFTSystem
from systems.grpo_system import GRPOSystem
from utils.text_encoder import TextEncoder
from data.finefs import EDGES, NUM_JOINTS

# ==================== Plot Color Definitions ====================
COL_COLORS = [
    np.array([0.1, 0.7, 0.3]),   # Green
    np.array([1.0, 0.5, 0.0]),   # Orange
    np.array([0.6, 0.2, 0.9]),   # Purple
    np.array([0.1, 0.6, 0.9]),   # Cyan
    np.array([0.9, 0.1, 0.4]),   # Pink
    np.array([0.7, 0.7, 0.1]),   # Olive
]

# ==================== Rendering Core (Generic Grid Visualization) ====================
def render_diversity_grid(trajectories, gt_data, swept_values, col_labels, num_variants, sweep_type, output_mp4, fps=30):
    print(f"🎥 Preparing to render diversity video: {output_mp4} ...")
    x_spacing = 0.55
    z_spacing = 0.70
    seq_len = gt_data.shape[0]
    
    vis_trajs = []
    
    for v_idx in range(num_variants):
        vis_trajs.append({
            "data": gt_data,
            "color": "royalblue",
            "alpha": 0.55,
            "linewidth": 1.5,
            "offset": np.array([-x_spacing, 0.0, v_idx * z_spacing])
        })
        
    num_cols = len(swept_values)
    for col_i, val in enumerate(swept_values):
        base_color = COL_COLORS[col_i % len(COL_COLORS)]
        for v_idx in range(num_variants):
            key = (col_i, v_idx)
            if key not in trajectories:
                continue
            pred_data = trajectories[key]
            
            color = tuple(base_color)
            alpha = 0.75 - (v_idx * 0.08)
            lw = 1.6 - (v_idx * 0.15)
            
            offset = np.array([col_i * x_spacing, 0.0, v_idx * z_spacing])
            vis_trajs.append({
                "data": pred_data,
                "color": color,
                "alpha": max(alpha, 0.3),
                "linewidth": max(lw, 0.8),
                "offset": offset
            })
            
    all_coords = np.concatenate([t["data"] + t["offset"] for t in vis_trajs], axis=0)
    min_v, max_v = np.min(all_coords, axis=(0, 1)), np.max(all_coords, axis=(0, 1))
    pad = 0.4
    x_lim = [min(-pad, min_v[0]) - 0.05, max(pad, max_v[0]) + 0.05]
    z_lim = [min(-pad, min_v[2]) - 0.05, max(pad, max_v[2]) + 0.05]
    y_lim = [-0.5, 0.5]
    lx, ly, lz = x_lim[1] - x_lim[0], z_lim[1] - z_lim[0], y_lim[1] - y_lim[0]
    
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((lx, ly, lz))
    
    legend_handles = [mlines.Line2D([], [], color="royalblue", label="GT Reference")]
    for col_i, label in enumerate(col_labels):
        c = tuple(COL_COLORS[col_i % len(COL_COLORS)])
        legend_handles.append(mlines.Line2D([], [], color=c, label=str(label)))
        
    col_label_y = z_lim[0] - 0.05
    col_label_z = y_lim[1]
    row_label_x = x_lim[0] - 0.05

    frames = []
    for t_idx in tqdm(range(seq_len), desc="Rendering frames"):
        ax.clear()
        ax.set_xlim(x_lim); ax.set_ylim(z_lim); ax.set_zlim(y_lim)
        ax.set_xlabel("X"); ax.set_ylabel("Depth (Z)"); ax.set_zlabel("Height (Y)")
        ax.view_init(elev=15, azim=-75)
        ax.dist = 7.5
        
        for x_val in np.arange(x_lim[0], x_lim[1], 0.5):
            ax.plot([x_val, x_val], [z_lim[0], z_lim[1]], [y_lim[0], y_lim[0]], color="lightgray", linestyle="--", linewidth=0.5)
        for z_val in np.arange(z_lim[0], z_lim[1], 0.5):
            ax.plot([x_lim[0], x_lim[1]], [z_val, z_val], [y_lim[0], y_lim[0]], color="lightgray", linestyle="--", linewidth=0.5)
            
        for traj in vis_trajs:
            pose = traj["data"][t_idx] + traj["offset"]
            xs, ys, zs = pose[:, 0], pose[:, 2], -pose[:, 1]
            ax.scatter(xs, ys, zs, c=[traj["color"]], s=8, alpha=traj["alpha"])
            for v1, v2 in EDGES:
                ax.plot([xs[v1], xs[v2]], [ys[v1], ys[v2]], [zs[v1], zs[v2]],
                        color=traj["color"], alpha=traj["alpha"], linewidth=traj["linewidth"])

        ax.text(-x_spacing, col_label_y, col_label_z, "GT", fontsize=8, ha="center", color="royalblue", fontweight="bold")
        for col_i, label in enumerate(col_labels):
            c = tuple(COL_COLORS[col_i % len(COL_COLORS)])
            ax.text(col_i * x_spacing, col_label_y, col_label_z, str(label), fontsize=7, ha="center", color=c, fontweight="bold")
            
        for v_idx in range(num_variants):
            ax.text(row_label_x, v_idx * z_spacing, col_label_z, f"Variant #{v_idx+1}", fontsize=7, ha="right", color="gray")

        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        fig.canvas.draw()
        
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8").reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())
        
    plt.close(fig)
    if frames:
        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        imageio.mimsave(output_mp4, frames, fps=fps)
        print(f"✅ Rendering complete! Saved to: {output_mp4}")

# ==================== Helper Utilities ====================
def load_hydra_config(exp_dir):
    cfg_path = os.path.join(exp_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Hydra config file not found: {cfg_path}")
    return OmegaConf.load(cfg_path)

def build_model_from_checkpoint(config, ckpt_path, device):
    print(f"📦 Loading weights: {ckpt_path}")
    is_rl = "kl_coef" in config or "rl" in config
    if is_rl:
        system = GRPOSystem.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
    else:
        system = SFTSystem.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
    
    system.to(device)
    system.eval()
    return system

def load_gt_motion(res_pk_path, config, window=0):
    with open(res_pk_path, "rb") as f:
        data = pickle.load(f)
    key = "pred_xyz_24_struct_global" if "pred_xyz_24_struct_global" in data else "pred_xyz_24_struct"
    xyz = data[key].astype(np.float32)
    
    input_n = config.dataset.get("input_n", 30)
    output_n = config.dataset.get("output_n", 40)
    total = input_n + output_n
    
    xyz = xyz[window : window + total]
    if len(xyz) < total:
        xyz = np.concatenate([xyz, np.repeat(xyz[-1:], total - len(xyz), axis=0)], axis=0)
    
    gt_pose = xyz.reshape(-1, NUM_JOINTS * 3)
    return gt_pose, input_n, output_n

def get_target_checkpoint(args, ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        raise ValueError(f"No .ckpt found under {ckpt_dir}")

    def get_epoch_or_step(path):
        name = os.path.basename(path)
        m_sft = re.search(r"sft_epoch=(\d+)", name)
        if m_sft: return int(m_sft.group(1))
        m_moe = re.search(r"moe_epoch=(\d+)", name)
        if m_moe: return int(m_moe.group(1))
        m_rl_ep = re.search(r"rl_epoch=(\d+)", name)
        if m_rl_ep: return int(m_rl_ep.group(1))
        m_rl_step = re.search(r"rl_step=(\d+)", name)
        if m_rl_step: return int(m_rl_step.group(1))
        m1 = re.search(r"epoch_1based=(\d+)", name)
        if m1: return int(m1.group(1))
        m2 = re.search(r"epoch=(\d+)", name)
        if m2: return int(m2.group(1)) + 1
        m3 = re.search(r"step=(\d+)", name)
        if m3: return int(m3.group(1))
        return -1

    target_step = getattr(args, "step", -1)
    if target_step != -1:
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            m_step = re.search(r"rl_step=(\d+)", name) or re.search(r"step=(\d+)", name)
            if m_step and int(m_step.group(1)) == target_step:
                return ckpt
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            if f"step={target_step}" in name or f"rl_step={target_step}" in name:
                return ckpt

    target_epoch = getattr(args, "epoch", -1)
    if target_epoch != -1:
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            m_ep = (re.search(r"sft_epoch=(\d+)", name) or 
                    re.search(r"moe_epoch=(\d+)", name) or 
                    re.search(r"epoch_1based=(\d+)", name) or 
                    re.search(r"epoch=(\d+)", name))
            if m_ep:
                val = int(m_ep.group(1))
                if "epoch=" in name and not any(k in name for k in ["sft_epoch", "moe_epoch", "epoch_1based", "rl-epoch"]):
                    val += 1
                if val == target_epoch:
                    return ckpt
        for ckpt in ckpts:
            name = os.path.basename(ckpt)
            if (f"epoch={target_epoch}" in name or 
                f"epoch_1based={target_epoch}" in name or 
                f"sft_epoch={target_epoch}" in name or 
                f"moe_epoch={target_epoch}" in name):
                return ckpt
        raise ValueError(f"No checkpoint matching Epoch/Step {target_epoch} found in {ckpt_dir}! Available files: {[os.path.basename(c) for c in ckpts]}")
    
    ckpts_sorted = sorted(ckpts, key=os.path.getmtime)
    return ckpts_sorted[-1]

def get_active_timesteps(sampling_mode, num_steps, num_discrete_steps=5, window_size=5, enabled=True):
    if not enabled:
        return None
    random_state = random.getstate()
    random.seed(42)
    try:
        if sampling_mode == "continuous":
            t_start = random.randint(window_size, num_steps - 1)
            active = list(range(t_start - window_size + 1, t_start + 1))
        elif sampling_mode == "discrete":
            active = random.sample(range(1, num_steps), num_discrete_steps)
        else:
            active = None
    finally:
        random.setstate(random_state)
    return active

# ==================== Main Computation and Statistics ====================
def main():
    parser = argparse.ArgumentParser(description="ReasonMotion RL/SFT diversity parameter sweep and analysis tool")
    parser.add_argument("--exp_dir", required=True, help="Training output folder (e.g. outputs/sft_xxx or outputs/rl_xxx)")
    parser.add_argument("--motion", default="/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk", help="Path to the pk file to test with")
    parser.add_argument("--text", default="triple", help="Text condition for inference")
    parser.add_argument("--epoch", type=int, default=-1, help="Which epoch checkpoint to load")
    parser.add_argument("--step", type=int, default=-1, help="Which step checkpoint to load")
    parser.add_argument("--sweep_type", choices=["std", "steps", "epoch"], default="std", help="Sweep type: std, steps, or epoch")
    parser.add_argument("--sampling_std", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.5], help="List of sampling_std values to test")
    parser.add_argument("--num_discrete_steps", nargs="+", type=int, default=[5, 10, 15, 20, 50], help="List of discrete exploration step counts to test")
    parser.add_argument("--epochs", nargs="+", type=int, default=[30, 50, 100, 200, 500], help="List of checkpoint epochs to compare")
    parser.add_argument("--steps", nargs="+", type=int, default=[900, 1800, 2700, 3600], help="List of checkpoint steps to compare")
    parser.add_argument("--sampling_mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for continuous mode")
    parser.add_argument("--no_delayed", action="store_true", help="Force-disable the delayed sub-trajectory")
    parser.add_argument("--num_variants", type=int, default=3, help="Number of variants to generate per parameter setting")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_hydra_config(args.exp_dir)
    text_encoder = TextEncoder(device=device)

    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    model = None
    if args.sweep_type != "epoch":
        ckpt_path = get_target_checkpoint(args, ckpt_dir)
        system = build_model_from_checkpoint(config, ckpt_path, device)
        model = system.model

    gt_pose, in_n, out_n = load_gt_motion(args.motion, config, window=0)
    gt_tensor = torch.tensor(gt_pose).unsqueeze(0).to(device)
    mask_tensor = torch.zeros_like(gt_tensor)
    mask_tensor[:, :in_n] = 1
    tp_tensor = torch.arange(gt_tensor.shape[1]).unsqueeze(0).float().to(device)
    gt_feed = {"pose": gt_tensor, "mask": mask_tensor, "timepoints": tp_tensor}
    
    text_emb = text_encoder([args.text])
    text_emb = (text_emb[0].to(device), text_emb[1].to(device))

    swept_values = []
    col_labels = []
    
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    is_rl_step = False
    if ckpts:
        name = os.path.basename(ckpts[0])
        is_rl_step = "step=" in name or "rl_step=" in name

    if args.sweep_type == "std":
        fixed_steps = args.num_discrete_steps[0]
        swept_values = args.sampling_std
        col_labels = [f"std={v}" for v in swept_values]
        print(f"🔍 Sweep type: sampling_std, values: {swept_values} | fixed active steps = {fixed_steps}")
    elif args.sweep_type == "steps":
        fixed_std = args.sampling_std[0]
        swept_values = args.num_discrete_steps
        col_labels = [f"steps={v}" for v in swept_values]
        print(f"🔍 Sweep type: active steps, values: {swept_values} | fixed std = {fixed_std}")
    else:
        if is_rl_step:
            swept_values = args.steps
            col_labels = [f"Step {v}" for v in swept_values]
            print(f"🔍 Sweep type: checkpoint steps, values: {swept_values} | fixed std = {args.sampling_std[0]}, active steps = {args.num_discrete_steps[0]}")
        else:
            swept_values = args.epochs
            col_labels = [f"Ep {v}" for v in swept_values]
            print(f"🔍 Sweep type: checkpoint epochs, values: {swept_values} | fixed std = {args.sampling_std[0]}, active steps = {args.num_discrete_steps[0]}")

    trajectories = {}
    diversity_stats = []

    print("\n--- 🚀 Starting diversity generation and statistics ---")
    
    for col_i, val in enumerate(swept_values):
        if args.sweep_type == "std":
            current_std = val
            current_steps = args.num_discrete_steps[0]
            current_model = model
        elif args.sweep_type == "steps":
            current_std = args.sampling_std[0]
            current_steps = val
            current_model = model
        else:
            current_std = args.sampling_std[0]
            current_steps = args.num_discrete_steps[0]
            
            class MockArgs:
                pass
            mock_args = MockArgs()
            if is_rl_step:
                mock_args.step = val
                mock_args.epoch = -1
            else:
                mock_args.epoch = val
                mock_args.step = -1
                
            ckpt_path = get_target_checkpoint(mock_args, ckpt_dir)
            system = build_model_from_checkpoint(config, ckpt_path, device)
            current_model = system.model
            
        current_model.sampling_std = current_std
        
        active_steps = get_active_timesteps(
            sampling_mode=args.sampling_mode,
            num_steps=current_model.num_steps,
            num_discrete_steps=current_steps,
            window_size=args.window_size,
            enabled=not args.no_delayed
        )
        
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        with torch.no_grad():
            final_samples, _, _, _ = current_model.sample_trajectory(
                text_emb, gt_feed, G=args.num_variants, active_timesteps=active_steps
            )
            
        variants = final_samples[0].detach().cpu()
        variants_TK = variants.permute(0, 2, 1).numpy()
        variants_struct = variants_TK.reshape(args.num_variants, -1, NUM_JOINTS, 3)
        
        for v_idx in range(args.num_variants):
            trajectories[(col_i, v_idx)] = variants_struct[v_idx]
            
        pred_variants = variants_struct[:, in_n:]
        
        std_per_joint_frame = np.std(pred_variants, axis=0)
        mean_std = np.mean(std_per_joint_frame) * 100.0
        max_std = np.max(std_per_joint_frame) * 100.0
        
        max_pairwise_dist = 0.0
        for i in range(args.num_variants):
            for j in range(i + 1, args.num_variants):
                dist = np.sqrt(np.sum((pred_variants[i] - pred_variants[j]) ** 2, axis=-1))
                max_pairwise_dist = max(max_pairwise_dist, np.max(dist))
        max_pairwise_dist *= 100.0
        
        print(f"📊 {col_labels[col_i]}: Mean Std = {mean_std:.3f} cm | Max Std = {max_std:.3f} cm | Max Pairwise Dist = {max_pairwise_dist:.3f} cm")
        diversity_stats.append({
            "label": col_labels[col_i],
            "mean_std_cm": mean_std,
            "max_std_cm": max_std,
            "max_pair_dist_cm": max_pairwise_dist
        })

    print("\n| Parameter Setting | Mean Std (cm) | Max Std (cm) | Max Pairwise Dist (cm) |")
    print("|---|---|---|---|")
    for stat in diversity_stats:
        print(f"| {stat['label']} | {stat['mean_std_cm']:.4f} | {stat['max_std_cm']:.4f} | {stat['max_pair_dist_cm']:.4f} |")

    out_dir = os.path.join(args.exp_dir, "visualize_suite")
    os.makedirs(out_dir, exist_ok=True)
    
    delayed_suffix = "all_steps" if args.no_delayed else f"{args.sampling_mode}_steps"
    output_mp4 = os.path.join(out_dir, f"sweep_{args.sweep_type}_{delayed_suffix}_s{args.seed}.mp4")
    
    gt_data_3d = gt_pose.reshape(-1, NUM_JOINTS, 3)
    render_diversity_grid(
        trajectories=trajectories,
        gt_data=gt_data_3d,
        swept_values=swept_values,
        col_labels=col_labels,
        num_variants=args.num_variants,
        sweep_type=args.sweep_type,
        output_mp4=output_mp4
    )

if __name__ == "__main__":
    main()
