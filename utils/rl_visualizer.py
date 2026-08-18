import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
import os
from typing import Dict, List, Tuple

# Use Agg backend to avoid display issues
matplotlib.use('Agg')

# SMPL_24 Edges (Same as FineFS)
EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18),
    (18, 20), (20, 22), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23)
]

class RLVisualizer:
    def __init__(self, output_dir: str, val_idx: int = 0, 
                 top_k: int = 3, bot_k: int = 3, viz_mode: str = "all", device: str = "cpu", joints: int = 24):
        self.output_dir = os.path.join(output_dir, "rl_policy_exploration")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.val_idx = val_idx
        self.top_k = top_k
        self.bot_k = bot_k
        self.viz_mode = viz_mode
        self.device = device
        self.joints = joints
        self.y_up = (joints == 17)  # H36M (17j) = Y-up, SMPL/FineFS (24j) & Boxing (25j) = Y-down
        
        if self.joints == 24:
            self.edges = EDGES
        elif self.joints == 25:
            from data.skeleton import EDGES_BODY25
            self.edges = EDGES_BODY25
        elif self.joints == 17:
            self.edges = [
                (0, 1), (1, 2), (2, 3),        # r-leg
                (0, 4), (4, 5), (5, 6),        # l-leg
                (0, 7), (7, 8), (8, 9), (9,10),# spine/head
                (8,11), (11,12), (12,13),      # l-arm
                (8,14), (14,15), (15,16)       # r-arm
            ]
        else:
            self.edges = []

        
        # Cache for the fixed validation sample
        self.fixed_sample: Dict = None

    def load_fixed_sample(self, val_dataset=None, target_pk_path="/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk"):
        """
        Loads and caches the specific sample from PK file directly for unified evaluation.
        Falls back to loading the first sample from dataset if the PK path is not found.
        """
        # Only use FineFS pk when joints match (24)
        if target_pk_path and self.joints == 24 and os.path.exists(target_pk_path):
            import pickle
            try:
                with open(target_pk_path, "rb") as f:
                    data = pickle.load(f)
                key = "pred_xyz_24_struct_global" if "pred_xyz_24_struct_global" in data else "pred_xyz_24_struct"
                xyz = data[key].astype(np.float32)
                
                input_n = 30
                output_n = 40
                total = input_n + output_n
                
                xyz = xyz[0 : total]
                if len(xyz) < total:
                    xyz = np.concatenate([xyz, np.repeat(xyz[-1:], total - len(xyz), axis=0)], axis=0)
                
                raw_pose_T_K = xyz.reshape(-1, self.joints * 3) # (T, K)
                
                pose = torch.tensor(raw_pose_T_K, device=self.device).float().unsqueeze(0) # (1, T, K)
                mask = torch.zeros_like(pose)
                mask[:, :input_n] = 1.0 # 1 for observed input
                tp = torch.arange(pose.shape[1], device=self.device).float().unsqueeze(0) # (1, T)
                
                self.fixed_sample = {
                    "pose": pose,          # (1, T, K)
                    "tp": tp,              # (1, T)
                    "mask": mask,          # (1, T, K)
                    "gen_mask": mask.clone(), # (1, T, K)
                    "motion_name": "quadruple",
                    "raw_pose_T_K": raw_pose_T_K
                }
                print(f"[RLVisualizer] Successfully loaded fixed PK sample from '{target_pk_path}'")
                return
            except Exception as e:
                print(f"[RLVisualizer] Failed to load sample from PK: {e}. Falling back to dataset.")
        
        if val_dataset is None:
            print("[RLVisualizer] Error: PK not found and val_dataset is None. Cannot load sample.")
            return

        # Search val_dataset for a jab punch sample
        chosen_idx = self.val_idx
        for i in range(len(val_dataset)):
            m_name = val_dataset[i].get("motion_name", "")
            if "jab" in m_name:
                chosen_idx = i
                break

        sample = val_dataset[chosen_idx]
        pose = torch.from_numpy(sample["pose"]).float().unsqueeze(0).to(self.device) # (1, T, K)
        tp = torch.from_numpy(sample["timepoints"]).float().unsqueeze(0).to(self.device)
        mask = torch.from_numpy(sample["mask"]).float().unsqueeze(0).to(self.device) # (1, T, K)
        
        self.fixed_sample = {
            "pose": pose,      # GT Pose (1, T, K)
            "tp": tp,
            "mask": mask,       # Original Mask (1, T, K)
            "gen_mask": mask.clone(), # Validation Mask (1 for input, 0 for output)
            "motion_name": sample["motion_name"],
            "raw_pose_T_K": sample["pose"]
        }
        print(f"[RLVisualizer] Loaded dataset jab sample idx {chosen_idx} ('{sample['motion_name']}')")


    @torch.no_grad()
    def run_epoch_viz(self, epoch: int, model, reward_model, text_encoder, current_std: float, num_variants: int, config=None, step: int = None):
        """
        Main function to run unified visualization for the epoch.
        """
        if self.fixed_sample is None:
            print("[RLVisualizer] Error: Fixed sample not loaded. Call load_fixed_sample() first.")
            return

        if step is not None:
            print(f"[RLVisualizer] Generating unified visualization for Step {step}...")
        else:
            print(f"[RLVisualizer] Generating unified visualization for Epoch {epoch}...")
        
        # [OOM Fix] Offload Reward Model to CPU to free up VRAM for Generation
        # We must restore it to GPU at the end.
        reward_model.cpu()
        torch.cuda.empty_cache()
        
        try:
            self._run_unified_exploration(epoch, model, reward_model, text_encoder, current_std, num_variants, config, step=step)
        except Exception as e:
            print(f"❌ Unified Viz Failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # [Restore] Always move Reward Model back to GPU for training
            print("[RLVisualizer] Restoring Reward Model to GPU...")
            reward_model.to(self.device)
            torch.cuda.empty_cache()

    def _prepare_common_inputs(self, text_encoder):
        # Prepare shared inputs
        pose = self.fixed_sample["pose"]
        tp = self.fixed_sample["tp"]
        gen_mask = self.fixed_sample["gen_mask"]
        motion_name = self.fixed_sample["motion_name"]

        # Text condition (Text Embedding)
        tok_emb, tok_mask = text_encoder([motion_name])
        tok_emb, tok_mask = tok_emb.to(self.device), tok_mask.to(self.device)
        text_cond = (tok_emb, tok_mask)
        
        feed = {"pose": pose, "mask": gen_mask, "timepoints": tp}
        return feed, text_cond, pose

    def _rank_and_render(self, epoch, mode_name, variants, reward_model, num_variants, pure_pred=None, step=None):
        """
        Ranks the generated variants by reward and prepares the rendering data.

        Args:
            pure_pred (Tensor, optional): The deterministic, noise-free prediction (Mean) trajectory, drawn in red. Shape: (T, K)
        """
        # The Reward Model needs GT data: (1, 1, T, J, 3)
        gt_pose = self.fixed_sample["pose"] # (1, T, K)
        gt_5d = gt_pose.reshape(1, 1, gt_pose.shape[1], self.joints, 3)

        # Handle variants: keep them on CPU to save memory
        if variants.device != torch.device("cpu"):
            variants = variants.detach().cpu()

        variants_TK = variants.permute(0, 2, 1) # (G, T, K)

        # Compute rewards in batches on CPU (to avoid OOM)
        rewards_list = []
        gt_batch_cpu = gt_5d.cpu()

        with torch.no_grad():
            for i in range(num_variants):
                # Extract a single variant: (1, 1, T, J, 3)
                v_batch = variants_TK[i:i+1].reshape(1, 1, variants_TK.shape[1], self.joints, 3)

                # Compute reward (both model and data on CPU)
                r, _ = reward_model(v_batch, gt_batch_cpu)
                rewards_list.append(r.item())
                
        rewards = np.array(rewards_list)
        std_val = rewards.std()
        if std_val < 1e-8:
            std_val = 1e-8
        advantages = (rewards - rewards.mean()) / std_val
        
        # Force top_k = 2, bot_k = 2 for unified mode
        top_k = 2
        bot_k = 2

        # Sort
        sorted_idx = np.argsort(rewards)
        best_idx = sorted_idx[-top_k:][::-1] # take the highest scores (Top 2)
        worst_idx = sorted_idx[:bot_k]       # take the lowest scores (Bot 2)

        # Collect trajectory data
        trajectories = []
        offset_step = 0.5 # X-axis offset applied to each trajectory, to avoid overlap

        # 1. Low-score variants (Bot K) - green shades, left to right: Bot-1 (leftmost, -1.0) -> Bot-2 (next, -0.5)
        bot_colors = ["lime", "yellowgreen"]
        for i, idx in enumerate(worst_idx):
            data = variants[idx].detach().permute(1, 0).cpu().numpy().reshape(-1, self.joints, 3)
            # worst_idx[0] (Bot-1) offset -1.0; worst_idx[1] (Bot-2) offset -0.5
            off = np.array([-(bot_k - i) * offset_step, 0, 0])
            color = bot_colors[i]
            alpha = 0.4 if i == 0 else 0.55
            lw = 1.0 if i == 0 else 1.25
            label = f"Bot-{i+1} (R:{rewards[idx]:.4f}, A:{advantages[idx]:+.2f})"
            trajectories.append({
                "data": data, "color": color, "alpha": alpha,
                "label": label, "linewidth": lw, "offset": off,
                "raw_reward": rewards[idx], "advantage": advantages[idx]
            })

        # 2. Ground-truth motion (GT) - blue (0.0)
        gt_data = self.fixed_sample["raw_pose_T_K"].reshape(-1, self.joints, 3)
        trajectories.append({
            "data": gt_data, "color": "blue", "alpha": 1.0, 
            "label": "GT (Ground Truth)", "linewidth": 2.0, "offset": np.array([0, 0, 0])
        })

        # 3. Pure Prediction - red (offset slightly to the right, 0.25)
        if pure_pred is not None:
            if pure_pred.device != torch.device("cpu"):
                pure_pred = pure_pred.detach().cpu()
            pure_data = pure_pred.numpy().reshape(-1, self.joints, 3)
            off = np.array([offset_step * 0.5, 0, 0])
            trajectories.append({
                "data": pure_data, "color": "red", "alpha": 0.9,
                "label": "Pure Pred (Mean)", "linewidth": 2.0, "offset": off
            })
        
        # 4. High-score variants (Top K) - green shades, left to right: Top-2 (next-to-rightmost, 0.5) -> Top-1 (rightmost, 1.0)
        top_colors = ["green", "darkgreen"] # Top-2 is green, Top-1 is darkgreen
        # To keep left-to-right spatial order, place Top-2 (i=1) first, then Top-1 (i=0)
        for i in reversed(range(top_k)):
            idx = best_idx[i]
            data = variants[idx].detach().permute(1, 0).cpu().numpy().reshape(-1, self.joints, 3)
            # Top-2 (i=1) offset 0.5; Top-1 (i=0) offset 1.0
            off = np.array([(top_k - i) * offset_step, 0, 0])
            color = top_colors[i]
            alpha = 0.65 if i == 1 else 0.8
            lw = 1.5 if i == 1 else 1.8
            label = f"Top-{i+1} (R:{rewards[idx]:.4f}, A:{advantages[idx]:+.2f})"
            trajectories.append({
                "data": data, "color": color, "alpha": alpha,
                "label": label, "linewidth": lw, "offset": off,
                "raw_reward": rewards[idx], "advantage": advantages[idx]
            })
            
        self.render_video(epoch, trajectories, gt_data, mode_name=mode_name, step=step)
    def _get_active_timesteps(self, config, epoch, num_steps: int):
        """Helper to get active timesteps for visualization based on config, reproducible per epoch."""
        if config is None or not hasattr(config, "rl") or not hasattr(config.rl, "delayed_sub_trajectory"):
            return None
            
        delayed = config.rl.delayed_sub_trajectory
        if not delayed.get("enabled", False):
            return None
            
        sampling_mode = delayed.get("sampling_mode", "discrete")
        window_size = delayed.get("window_size", 5)
        num_discrete_steps = delayed.get("num_discrete_steps", 5)
        
        # Determine seed based on epoch for reproducibility of the frame selection
        epoch_num = int(str(epoch).split('_')[0]) if isinstance(epoch, str) else epoch
        
        import random
        # Save random state
        state = random.getstate()
        random.seed(12345 + epoch_num)
        
        try:
            if sampling_mode == "continuous":
                t_start = random.randint(window_size, num_steps - 1)
                active = list(range(t_start - window_size + 1, t_start + 1))
            elif sampling_mode == "discrete":
                active = random.sample(range(1, num_steps), num_discrete_steps)
            else:
                active = None
        finally:
            # Restore random state
            random.setstate(state)
            
        return active

    def _run_unified_exploration(self, epoch, model, reward_model, text_encoder, current_std, num_variants, config, step=None):
        """
        Runs unified visualization containing:
        1. Blue line: Ground Truth
        2. Red line: Deterministic DDIM trajectory (z=0 mean)
        3. Four green variants: Top-2 highest reward (darker green) and Bot-2 lowest reward (lighter green)
        faithful to config.rl.delayed_sub_trajectory settings.
        """
        torch.cuda.empty_cache()
        epoch_num = int(str(epoch).split('_')[0]) if isinstance(epoch, str) else epoch
        torch.manual_seed(123 + epoch_num)
        
        feed, text_cond, _ = self._prepare_common_inputs(text_encoder)
        
        # 1. Deterministic DDIM prediction (sample=False, z=0)
        p_pure = model.evaluate(feed, 1, text_embedding=text_cond, sample=False)[0]
        pure_pred = p_pure[0, 0].cpu() # (K, L)
        pure_pred = pure_pred.permute(1, 0) # (T, K)
        
        # 2. Get active timesteps based on training config
        active_timesteps = self._get_active_timesteps(config, epoch, model.num_steps)
        print(f"[RLVisualizer] Unified Exploration using active_timesteps: {active_timesteps}")
        
        # 3. Sample stochastic variants
        final_samples, _, _, _ = model.sample_trajectory(
            text_cond, feed, G=num_variants, active_timesteps=active_timesteps
        )
        variants = final_samples[0] # (G, K, L)
        
        # 4. Rank and Render
        self._rank_and_render(epoch, "unified_exploration", variants, reward_model, num_variants, pure_pred=pure_pred, step=step)

    def render_video(self, epoch, trajectories, gt_pose, mode_name="default", step=None):
        """
        Renders the accumulated trajectories into a video.
        """
        frames = []
        seq_len = gt_pose.shape[0] # Should be 90 (30 input + 40 output + padding?) or 70
        input_n = 30
        
        # [Fix] Ensure generated variants start from GT input (t=0 to 30) for visual continuity
        # The diffusion model might output slightly different values even for conditioned frames if not enforced.
        # But `model.impute` uses `cond_mask` to enforce observed data. 
        # If it "jumps" at t=30, it might be that `gen_mask` was zeros?
        # In `load_fixed_sample`, gen_mask = torch.zeros_like(mask).
        # This means generation is UNCONDITIONED (In-painting mask is all 0 means generate everything?).
        # Wait, `impute` logic uses `cond_mask` to overwrite `total_input`.
        # If `gen_mask` is all zeros, then NOTHING is preserved?
        # Ah! `model.impute` code:
        # `model_input = (1 - cond_mask) * noisy_data + cond_mask * observed_data`
        # If `cond_mask` is 0, it uses `noisy_data` (generated).
        # So we definitely want to visualize the model generating FROM scratch? 
        # USER said: "the last 40 frames blow up". Before that should be GT?
        # If we pass `gen_mask` as all zeros, the model regenerates 0-30 too?
        # Let's check `load_fixed_sample`:
        # `gen_mask = torch.zeros_like(mask)`
        # `mask` from dataset is typically 1 for observed (0-30) and 0 for unobserved (30-70).
        # IF we want conditioned generation, we should use the original `mask`!
        # `feed` uses `gen_mask`.
        
        # Setup Figure once
        fig = plt.figure(figsize=(12, 8)) # Wider for legend
        ax = fig.add_subplot(111, projection='3d') 
        # But user also mentioned "draw the axes and scale".
        # Let's stick to 2D for clarity if user wants "XY plane front view",
        # BUT 3D is better for "slightly tilted".
        # User said: "straight-on (XY plane) + slightly tilted 3D view" -> 3D Plot with fixed ViewInit.
        
        # Determine strict bounds to prevent camera jumping
        # Use GT to define "normal" bounds, maybe expand slightly
        # [Fix] Include offset in bounds calculation
        all_coords_list = []
        for t in trajectories:
            # Apply offset to data for bounds calculation
            # t["data"] is (T, 24, 3)
            # t["offset"] is (3,)
            adjusted_data = t["data"] + t["offset"]
            all_coords_list.append(adjusted_data)
        
        all_coords = np.concatenate(all_coords_list, axis=0) # (Total_Frames, J, 3)
        min_vals = np.min(all_coords, axis=(0, 1))
        max_vals = np.max(all_coords, axis=(0, 1))
        
        # Force bounds to be at least [-1, 1] for stability if motions are small
        bound = 1.0
        x_lim = [min(-bound, min_vals[0]), max(bound, max_vals[0])]
        if self.y_up:
            y_lim = [min(-bound, min_vals[1]), max(bound, max_vals[1])]
        else:
            y_lim = [min(-bound, -max_vals[1]), max(bound, -min_vals[1])] # Flipped Y
        z_lim = [min(-bound, min_vals[2]), max(bound, max_vals[2])]

        print(f"[RLVisualizer] Rendering {seq_len} frames for mode '{mode_name}'...")
        
        for t in range(seq_len):
            ax.clear()
            
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim) # Y is flipped
            ax.set_zlim(y_lim) # Height (Visual Z)
            ax.set_ylim(z_lim) # Depth (Visual Y)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Z (Depth)')
            ax.set_zlabel('Y (Height)')
            ax.set_title(f"Epoch {epoch} | Frame {t} | Mode: {mode_name}")
            
            # View Init (Front-ish but tilted)
            ax.view_init(elev=10, azim=-90) # azim=-90 puts X axis horizontal, Y axis depth.
            
            # Plot Logic
            for traj in trajectories:
                # Get current pose
                pose = traj["data"][t] # (J, 3)
                
                # Apply Offset (in Data Coordinates)
                pose = pose + traj["offset"]
                
                # Map to Visual Coordinates
                # Plot X = Data X, Plot Y = Data Z (depth), Plot Z = vertical
                xs = pose[:, 0]
                ys = pose[:, 2]
                zs = pose[:, 1] if self.y_up else -pose[:, 1]
                
                # Scatter Joints
                ax.scatter(xs, ys, zs, c=traj["color"], s=10, alpha=traj["alpha"])
                
                # Draw Bones
                for (v1, v2) in self.edges:
                    x_pair = [xs[v1], xs[v2]]
                    y_pair = [ys[v1], ys[v2]]
                    z_pair = [zs[v1], zs[v2]]
                    ax.plot(x_pair, y_pair, z_pair, color=traj["color"], 
                            alpha=traj["alpha"], linewidth=traj["linewidth"])

                # Draw floating text for variant's reward/advantage above the skeleton
                if "raw_reward" in traj and "advantage" in traj:
                    text_x = np.mean(xs)
                    text_y = np.mean(ys)
                    text_z = np.max(zs) + 0.15
                    ax.text(text_x, text_y, text_z, 
                            f"R:{traj['raw_reward']:.4f}\nA:{traj['advantage']:+.2f}", 
                            color=traj["color"], fontsize=7, ha='center', va='bottom', fontweight='bold')

            # Dynamic Custom Legend
            import matplotlib.lines as mlines
            legend_elements = []
            for traj in trajectories:
                legend_elements.append(
                    mlines.Line2D([], [], color=traj["color"], alpha=traj["alpha"], label=traj["label"])
                )
            ax.legend(handles=legend_elements, loc='upper right')

            # Save frame
            fig.canvas.draw()
            # Compatible with Matplotlib 3.x Agg backend
            s, (width, height) = fig.canvas.print_to_buffer()
            image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            image = image[:, :, :3] # Drop Alpha channel to get RGB
            frames.append(image)
            
        plt.close(fig)
        
        # Save Video
        if step is not None:
            save_path = os.path.join(self.output_dir, f"step_{step}_{mode_name}.mp4")
        else:
            save_path = os.path.join(self.output_dir, f"epoch_{epoch}_{mode_name}.mp4")
        render_fps = 60 if self.joints == 25 else 30
        imageio.mimsave(save_path, frames, fps=render_fps)
        print(f"[RLVisualizer] Saved video to {save_path}")
