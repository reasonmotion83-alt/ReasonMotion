import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
ReasonMotion Visualization and Testing Suite (Visualizer Suite)
This script generates 3D skeleton videos across various dimensions, supporting SFT, MoE, and RL models.

Usage examples:

1. [infer mode] - The purest visualization: generates just the 1 specified motion and does a side-by-side comparison with GT.
   python scripts/visualize.py --exp_dir outputs/sft_finefs_2026-07-13_12-24 --mode infer --motion /home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk --texts triple

2. [grid mode] - Good for tracking RL evolution: loads all checkpoints and plots an evolution grid with "X-axis = training progress" x "Z-axis = different prompts".
   python scripts/visualize.py --exp_dir outputs/sft_finefs_2026-07-13_12-24 --mode grid --motion /home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk --texts single double triple quadruple

3. [cfg_sweep mode] - Finding the optimal guidance strength: for the best checkpoint, tests different cfg_scale values (0.0~5.0) to observe the collapse point.
   python scripts/visualize.py --exp_dir outputs/sft_finefs_2026-07-13_12-24 --mode cfg_sweep --motion /home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk --texts triple

4. [diversity mode] - Diversity and stability test: fixes the prompt and generates with 5 different random seeds for a side-by-side comparison (to catch bad cases and mode collapse).
   python scripts/visualize.py --exp_dir outputs/sft_finefs_2026-07-13_12-24 --mode diversity --motion /home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk --texts triple
"""
import glob
import re
import argparse
import pickle
import numpy as np
import torch
import imageio
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
from data.finefs import EDGES as EDGES_FINEFS, NUM_JOINTS as NUM_JOINTS_FINEFS
from data.skeleton import EDGES_BODY25

# ==================== Plotting Constants ====================
TEXT_BASE_COLORS = {
    # Figure Skating
    "single":    np.array([1.0, 0.20, 0.20]),   # Red
    "double":    np.array([1.0, 0.55, 0.00]),   # Orange
    "triple":    np.array([0.55, 0.20, 1.00]),  # Purple
    "quadruple": np.array([0.10, 0.75, 0.20]),  # Green
    # Boxing Punches
    "forehand_jab":      np.array([1.0, 0.20, 0.20]),
    "backhand_jab":      np.array([1.0, 0.55, 0.00]),
    "forehand_hook":     np.array([0.55, 0.20, 1.00]),
    "backhand_hook":     np.array([0.10, 0.75, 0.20]),
    "forehand_uppercut": np.array([0.20, 0.60, 1.00]),
    "backhand_uppercut": np.array([0.90, 0.20, 0.60]),
    "neutral":           np.array([0.50, 0.50, 0.50]),
}


# ==================== Rendering Core (Generic Grid Visualization) ====================
def render_3d_grid_video(trajectories, gt_data, texts, col_labels, output_mp4, fps=30):
    print(f"🎥 Preparing to render video: {output_mp4} ...")
    x_spacing = 0.55
    z_spacing = 0.70
    seq_len = gt_data.shape[0]
    
    vis_trajs = []
    
    for row_i, txt in enumerate(texts):
        vis_trajs.append({
            "data": gt_data,
            "color": "royalblue",
            "alpha": 0.45,
            "linewidth": 1.5,
            "offset": np.array([0.0, 0.0, row_i * z_spacing])
        })
        
    num_cols = len(col_labels)
    for row_i, txt in enumerate(texts):
        base_c = TEXT_BASE_COLORS.get(txt, np.array([0.5, 0.5, 0.5]))
        for col_i, col_label in enumerate(col_labels):
            if (col_i, txt) not in trajectories:
                continue
            pred_data = trajectories[(col_i, txt)]
            
            ratio = col_i / max(num_cols - 1, 1)
            brightness = 0.35 + 0.65 * ratio
            color = tuple(np.clip(base_c * brightness, 0, 1))
            alpha = 0.45 + 0.55 * ratio
            
            offset = np.array([(col_i + 1) * x_spacing, 0.0, row_i * z_spacing])
            vis_trajs.append({
                "data": pred_data,
                "color": color,
                "alpha": alpha,
                "linewidth": 1.5,
                "offset": offset
            })
            
    all_coords = np.concatenate([t["data"] + t["offset"] for t in vis_trajs], axis=0)
    min_v, max_v = np.min(all_coords, axis=(0, 1)), np.max(all_coords, axis=(0, 1))
    pad = 0.5
    x_lim = [min(-pad, min_v[0]) - 0.1, max(pad, max_v[0]) + 0.1]
    z_lim = [min(-pad, min_v[2]) - 0.1, max(pad, max_v[2]) + 0.1]
    y_lim = [-0.5, 0.5]
    lx, ly, lz = x_lim[1] - x_lim[0], z_lim[1] - z_lim[0], y_lim[1] - y_lim[0]
    
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((lx, ly, lz))
    
    legend_handles = [mlines.Line2D([], [], color="royalblue", label="GT")]
    for txt in texts:
        c = tuple(TEXT_BASE_COLORS.get(txt, np.array([0.5, 0.5, 0.5])))
        legend_handles.append(mlines.Line2D([], [], color=c, label=txt.capitalize()))
        
    col_label_y = z_lim[0] - 0.05
    col_label_z = y_lim[1]
    row_label_x = x_lim[0] - 0.05

    frames = []
    for t_idx in tqdm(range(seq_len), desc="Rendering frames"):
        ax.clear()
        ax.set_xlim(x_lim); ax.set_ylim(z_lim); ax.set_zlim(y_lim)
        ax.set_xlabel("X"); ax.set_ylabel("Depth"); ax.set_zlabel("Height")
        ax.view_init(elev=20, azim=-60)
        ax.dist = 7.0
        
        for traj in vis_trajs:
            pose = traj["data"][t_idx] + traj["offset"]
            xs, ys, zs = pose[:, 0], pose[:, 2], -pose[:, 1]
            ax.scatter(xs, ys, zs, c=[traj["color"]], s=12, alpha=traj["alpha"])
            edges = EDGES_BODY25 if pose.shape[0] == 25 else EDGES_FINEFS
            for v1, v2 in edges:
                if v1 < pose.shape[0] and v2 < pose.shape[0]:
                    ax.plot([xs[v1], xs[v2]], [ys[v1], ys[v2]], [zs[v1], zs[v2]],
                            color=traj["color"], alpha=traj["alpha"], linewidth=traj["linewidth"])


        ax.text(0, col_label_y, col_label_z, "GT", fontsize=7, ha="center", color="royalblue", fontweight="bold")
        for col_i, label in enumerate(col_labels):
            ax.text((col_i + 1) * x_spacing, col_label_y, col_label_z, str(label), fontsize=6, ha="center", color="gray")
        for row_i, txt in enumerate(texts):
            c = tuple(TEXT_BASE_COLORS.get(txt, np.array([0.5, 0.5, 0.5])))
            ax.text(row_label_x, row_i * z_spacing, col_label_z, txt.capitalize(), fontsize=7, ha="right", color=c, fontweight="bold")

        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)
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
    ds_name = config.get("dataset", {}).get("name", "").lower()
    if ds_name == "h36m":
        target_dim = config.get("dataset", {}).get("joints", 17) * 3
    elif ds_name == "boxing":
        target_dim = 25 * 3
    else:
        target_dim = 24 * 3

    is_rl = "kl_coef" in config
    if is_rl:
        system = GRPOSystem(config=config, target_dim=target_dim)
    else:
        system = SFTSystem(config=config, target_dim=target_dim)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    system.load_state_dict(ckpt["state_dict"], strict=False)

    system.to(device)
    system.eval()
    return system


def load_gt_motion(res_pk_path, config, window=0, sample_idx=0):
    input_n = config.dataset.get("input_n", 30)
    output_n = config.dataset.get("output_n", 40)
    total = input_n + output_n

    if res_pk_path and os.path.exists(res_pk_path) and res_pk_path.endswith(".pk"):
        with open(res_pk_path, "rb") as f:
            data = pickle.load(f)
        key = "pred_xyz_24_struct_global" if "pred_xyz_24_struct_global" in data else "pred_xyz_24_struct"
        xyz = data[key].astype(np.float32)
        xyz = xyz[window : window + total]
        if len(xyz) < total:
            xyz = np.concatenate([xyz, np.repeat(xyz[-1:], total - len(xyz), axis=0)], axis=0)
        num_joints = xyz.shape[1] if xyz.ndim == 3 else 24
        gt_pose = xyz.reshape(-1, num_joints * 3)
    else:
        # Load sample directly from dataset
        ds_name = config.get("dataset", {}).get("name", "finefs").lower()
        from train import build_dataset
        dataset, _, _ = build_dataset(config, split=1)
        if sample_idx >= len(dataset):
            sample_idx = len(dataset) - 1
        sample = dataset[sample_idx]
        gt_pose = sample["pose"] # (T, K)
        if hasattr(gt_pose, "numpy"):
            gt_pose = gt_pose.numpy()

    return gt_pose, input_n, output_n


# ==================== Main Functional Modes ====================
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

def parse_epoch_num(path):
    name = os.path.basename(path)
    m_step = re.search(r"rl_step=(\d+)", name) or re.search(r"step=(\d+)", name)
    if m_step:
        return f"step{m_step.group(1)}"
    m_sft = re.search(r"sft_epoch=(\d+)", name)
    if m_sft:
        return f"ep{m_sft.group(1)}"
    m_moe = re.search(r"moe_epoch=(\d+)", name)
    if m_moe:
        return f"ep{m_moe.group(1)}"
    m1 = re.search(r"epoch_1based=(\d+)", name)
    if m1:
        return f"ep{m1.group(1)}"
    m2 = re.search(r"epoch=(\d+)", name)
    if m2:
        return f"ep{int(m2.group(1)) + 1}"
    return "epunknown"

def run_grid_mode(args, config, device, text_encoder, gt_feed):
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    
    def get_epoch_or_step(path):
        name = os.path.basename(path)
        m_step = re.search(r"rl_step=(\d+)", name) or re.search(r"step=(\d+)", name)
        if m_step:
            return int(m_step.group(1))
        m_sft = re.search(r"sft_epoch=(\d+)", name)
        if m_sft: return int(m_sft.group(1))
        m_moe = re.search(r"moe_epoch=(\d+)", name)
        if m_moe: return int(m_moe.group(1))
        m1 = re.search(r"epoch_1based=(\d+)", name)
        if m1: return int(m1.group(1))
        m2 = re.search(r"epoch=(\d+)", name)
        if m2: return int(m2.group(1)) + 1
        return -1
        
    ckpts = sorted(ckpts, key=get_epoch_or_step)
    
    if not ckpts:
        raise ValueError(f"No .ckpt found under {ckpt_dir}")

    # Filter by --steps if specified
    if getattr(args, "steps", None) and len(args.steps) > 0:
        filtered_ckpts = []
        for step_val in args.steps:
            for c in ckpts:
                if get_epoch_or_step(c) == step_val:
                    filtered_ckpts.append(c)
                    break
        if filtered_ckpts:
            ckpts = filtered_ckpts
    elif getattr(args, "max_checkpoints", -1) > 0 and len(ckpts) > args.max_checkpoints:
        # Uniformly sample max_checkpoints from sorted checkpoints
        indices = np.linspace(0, len(ckpts) - 1, args.max_checkpoints, dtype=int)
        ckpts = [ckpts[i] for i in indices]
        
    print(f"🔍 After filtering, {len(ckpts)} checkpoints will be evaluated.")
    col_labels = []
    for c in ckpts:
        name = os.path.basename(c)
        prefix = "Step" if ("step" in name or "rl_step" in name) else "Ep"
        col_labels.append(f"{prefix} {get_epoch_or_step(c)}")
    
    trajectories = {}
    for col_i, ckpt in enumerate(ckpts):
        system = build_model_from_checkpoint(config, ckpt, device)
        for txt in args.texts:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            text_emb = text_encoder([txt])
            with torch.no_grad():
                out, _, _, _ = system.model.evaluate(gt_feed, n_samples=1, text_embedding=text_emb)
            
            p = out[0, 0].cpu().numpy().transpose(1, 0)
            num_j = p.shape[1] // 3
            trajectories[(col_i, txt)] = p.reshape(-1, num_j, 3)
            
    return trajectories, col_labels, None


def run_cfg_sweep_mode(args, config, device, text_encoder, gt_feed):
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    ckpt_path = get_target_checkpoint(args, ckpt_dir)
    
    system = build_model_from_checkpoint(config, ckpt_path, device)
    cfg_scales = [0.0, 1.0, 2.0, 3.0, 5.0]
    col_labels = [f"CFG {s}" for s in cfg_scales]
    
    trajectories = {}
    for col_i, scale in enumerate(cfg_scales):
        if hasattr(system, 'config'):
            OmegaConf.set_readonly(system.config, False)
            system.config.training.cfg_scale_eval = scale
        
        for txt in args.texts:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            text_emb = text_encoder([txt])
            with torch.no_grad():
                out, _, _, _ = system.model.evaluate(gt_feed, n_samples=1, text_embedding=text_emb)
            p = out[0, 0].cpu().numpy().transpose(1, 0)
            num_j = p.shape[1] // 3
            trajectories[(col_i, txt)] = p.reshape(-1, num_j, 3)
            
    return trajectories, col_labels, ckpt_path

def run_diversity_mode(args, config, device, text_encoder, gt_feed):
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    ckpt_path = get_target_checkpoint(args, ckpt_dir)
    
    system = build_model_from_checkpoint(config, ckpt_path, device)
    seeds = [args.seed, args.seed+1, args.seed+2, args.seed+3, args.seed+4]
    col_labels = [f"Seed {s}" for s in seeds]
    
    trajectories = {}
    for col_i, sd in enumerate(seeds):
        for txt in args.texts:
            torch.manual_seed(sd)
            np.random.seed(sd)
            text_emb = text_encoder([txt])
            with torch.no_grad():
                out, _, _, _ = system.model.evaluate(gt_feed, n_samples=1, text_embedding=text_emb)
            p = out[0, 0].cpu().numpy().transpose(1, 0)
            num_j = p.shape[1] // 3
            trajectories[(col_i, txt)] = p.reshape(-1, num_j, 3)
            
    return trajectories, col_labels, ckpt_path

def run_infer_mode(args, config, device, text_encoder, gt_feed):
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    ckpt_path = get_target_checkpoint(args, ckpt_dir)
    
    system = build_model_from_checkpoint(config, ckpt_path, device)
    col_labels = ["Prediction"]
    
    trajectories = {}
    for txt in args.texts:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        text_emb = text_encoder([txt])
        with torch.no_grad():
            out, _, _, _ = system.model.evaluate(gt_feed, n_samples=1, text_embedding=text_emb)
        p = out[0, 0].cpu().numpy().transpose(1, 0)
        num_j = p.shape[1] // 3
        trajectories[(0, txt)] = p.reshape(-1, num_j, 3)
            
    return trajectories, col_labels, ckpt_path

def main():
    parser = argparse.ArgumentParser(description="ReasonMotion all-purpose visualization detective")
    parser.add_argument("--exp_dir", required=True, help="Hydra training output folder (e.g. outputs/sft_xxx)")
    parser.add_argument("--mode", type=str, choices=["grid", "cfg_sweep", "diversity", "infer"], default="infer")
    parser.add_argument("--motion", type=str, default=None, help="Path to the pk file to test with (if omitted, automatically loads a random sample from the Boxing/FineFS dataset)")
    parser.add_argument("--texts", nargs="+", default=None, help="List of prompts (defaults to a random motion type or default motion)")
    parser.add_argument("--window", type=int, default=0, help="Starting point for the skeleton clip (sliding window)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epoch", type=int, default=-1, help="Which epoch to load (default -1 loads the latest epoch)")
    parser.add_argument("--step", type=int, default=-1, help="Which step to load (RL-only, default -1 loads the latest step)")
    parser.add_argument("--steps", type=int, nargs="+", default=None, help="Specify multiple steps (e.g. --steps 60 120 300 720)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to randomly draw from the dataset when --motion is not specified (e.g. 20)")
    parser.add_argument("--filter_punches", action="store_true", help="Preferentially filter for genuine punch motions (excluding the neutral pose)")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_hydra_config(args.exp_dir)
    ds_name = config.get("dataset", {}).get("name", "finefs").lower()
    text_encoder = TextEncoder(device=device)

    # Determine the sample source
    dataset_samples = []
    if args.motion and os.path.exists(args.motion) and args.motion.endswith(".pk"):
        dataset_samples.append({"path": args.motion, "label": None, "idx": 0})
    else:
        from train import build_dataset
        print(f"📦 Loading dataset ({ds_name}) for sample drawing (num_samples={args.num_samples})...")
        val_dataset, _, _ = build_dataset(config, split=1)
        np.random.seed(args.seed)

        all_indices = list(range(len(val_dataset)))
        if args.filter_punches:
            punch_indices = [i for i in all_indices if val_dataset[i].get("motion_name", "neutral") != "neutral"]
            print(f"🥊 Punch motion filtering enabled: drawing samples from {len(punch_indices)} genuine punch samples...")
            candidate_indices = punch_indices if len(punch_indices) >= args.num_samples else all_indices
        else:
            candidate_indices = all_indices
            
        chosen_indices = np.random.choice(candidate_indices, size=min(args.num_samples, len(candidate_indices)), replace=False)
        for idx in chosen_indices:
            item = val_dataset[idx]
            m_name = item.get("motion_name", "neutral")
            dataset_samples.append({"path": None, "label": m_name, "idx": idx, "item": item})


    out_dir = os.path.join(args.exp_dir, "visualize_suite")
    os.makedirs(out_dir, exist_ok=True)

    for sample_info in dataset_samples:
        s_idx = sample_info["idx"]
        motion_label = sample_info["label"]
        
        gt_pose, in_n, out_n = load_gt_motion(sample_info["path"], config, window=args.window, sample_idx=s_idx)
        num_j = gt_pose.shape[1] // 3
        gt_data_3d = gt_pose.reshape(-1, num_j, 3)

        gt_tensor = torch.tensor(gt_pose).unsqueeze(0).to(device)
        mask_tensor = torch.zeros_like(gt_tensor)
        mask_tensor[:, :in_n] = 1
        tp_tensor = torch.arange(gt_tensor.shape[1]).unsqueeze(0).float().to(device)
        gt_feed = {"pose": gt_tensor, "mask": mask_tensor, "timepoints": tp_tensor}
        if sample_info.get("item") and "motion_name" in sample_info["item"]:
            gt_feed["motion_name"] = [sample_info["item"]["motion_name"]]

        # Set the prompts
        if args.texts is None or len(args.texts) == 0:
            if motion_label:
                texts = [motion_label]
            elif ds_name == "boxing":
                texts = ["forehand_jab"]
            else:
                texts = ["triple"]
        else:
            texts = args.texts
        args.texts = texts

        print(f"\n🚀 Launching mode: {args.mode.upper()} [Sample Index: {s_idx}, Prompt: {texts}]")
        ckpt_path = None
        if args.mode == "grid":
            trajectories, col_labels, _ = run_grid_mode(args, config, device, text_encoder, gt_feed)
        elif args.mode == "cfg_sweep":
            trajectories, col_labels, ckpt_path = run_cfg_sweep_mode(args, config, device, text_encoder, gt_feed)
        elif args.mode == "diversity":
            trajectories, col_labels, ckpt_path = run_diversity_mode(args, config, device, text_encoder, gt_feed)
        elif args.mode == "infer":
            trajectories, col_labels, ckpt_path = run_infer_mode(args, config, device, text_encoder, gt_feed)
            
        sample_tag = f"_sample{s_idx:03d}_{texts[0]}" if len(dataset_samples) > 1 else ""
        if args.mode == "grid":
            out_mp4 = os.path.join(out_dir, f"{args.mode}{sample_tag}_w{args.window}_s{args.seed}.mp4")
            pk_path = os.path.join(out_dir, f"{args.mode}{sample_tag}_w{args.window}_s{args.seed}.pk")
        else:
            epoch_str = parse_epoch_num(ckpt_path)
            out_mp4 = os.path.join(out_dir, f"{args.mode}_{epoch_str}{sample_tag}_w{args.window}_s{args.seed}.mp4")
            pk_path = os.path.join(out_dir, f"{args.mode}_{epoch_str}{sample_tag}_w{args.window}_s{args.seed}.pk")
        
        render_3d_grid_video(trajectories, gt_data_3d, texts, col_labels, out_mp4)
        
        with open(pk_path, "wb") as f:
            pickle.dump({"trajectories": trajectories, "col_labels": col_labels, "texts": texts}, f)
        print(f"💾 Numpy arrays backed up to: {pk_path}")

if __name__ == "__main__":
    main()

