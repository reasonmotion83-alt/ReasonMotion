import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import torch
import pytorch_lightning as pl

# ----------------- Step Correction Loggers ----------------- #
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

class StepCorrectedTensorBoardLogger(TensorBoardLogger):
    def log_metrics(self, metrics, step=None):
        if step is not None:
            step = step + 1
        super().log_metrics(metrics, step=step)

class StepCorrectedCSVLogger(CSVLogger):
    def log_metrics(self, metrics, step=None):
        if step is not None:
            step = step + 1
        super().log_metrics(metrics, step=step)


# ----------------- Generic plotting script ----------------- #
def plot_3d_motion(pose_seq, edges, save_path=None, pause=0.001, fps=30, y_up=None):
    """
    Renders a 3D skeleton sequence and saves it as an mp4, or displays it directly.
    pose_seq : a (T, J, 3) or (T, J*3) numpy array or tensor
    edges    : list of tuple (i, j) skeleton connectivity
    save_path: absolute or relative output path for the mp4 (e.g. output/vis.mp4)
    y_up     : True = Y axis points up (H36M), False = Y axis points down (FineFS/SMPL)
               None = auto-detect (compares the joint heights with max/min Y in the first frame)
    """
    if isinstance(pose_seq, torch.Tensor):
        pose_seq = pose_seq.detach().cpu().numpy()
        
    T = pose_seq.shape[0]
    if pose_seq.ndim == 2:
        J = pose_seq.shape[1] // 3
        xyz = pose_seq.reshape(T, J, 3)
    else:
        J = pose_seq.shape[1]
        xyz = pose_seq

    # Auto-detect coordinate system orientation:
    # Y-up (H36M): head Y > foot Y, most joints (torso+arms) are above the pelvis
    # Y-down (FineFS/SMPL): head Y < foot Y, most joints are below the pelvis
    if y_up is None:
        y_vals = xyz[0, :, 1]
        y_center = xyz[0, 0, 1]  # joint 0 = pelvis
        # Use the median: in a standing body, most joints (torso+both arms+head) are above the pelvis
        # Y-up → median > center; Y-down → median < center
        y_up = np.median(y_vals) > y_center

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])  # to keep the aspect ratio correct
    ax.set_zlim([-0.6, 0.6])
    ax.set_axis_off()
    
    # Draw a simple floor
    floor = 0.8
    ax.plot_surface(*np.meshgrid(np.linspace(-floor, floor, 2),
                                 np.linspace(-floor, floor, 2)),
                    np.full((2, 2), -0.6), color='lightgray', alpha=0.15)

    lines = [ax.plot([], [], [], lw=2, c='blue')[0] for _ in edges]
    scat = ax.scatter([], [], [], s=20, c='blue')

    def update_frame(t):
        pts = xyz[t]
        
        # Axis mapping: matplotlib 3D's Z axis is the vertical direction on screen
        # X_plot = X,  Y_plot = Z(depth),  Z_plot = ±Y(vertical)
        xs = pts[:, 0]
        ys = pts[:, 2]
        zs = pts[:, 1] if y_up else -pts[:, 1]

        scat._offsets3d = (xs, ys, zs)
        for k, (i, j) in enumerate(edges):
            if i >= J or j >= J:
                lines[k].set_data([], [])
                lines[k].set_3d_properties([])
                continue
            
            lines[k].set_data(np.array([xs[i], xs[j]]), np.array([ys[i], ys[j]]))
            lines[k].set_3d_properties(np.array([zs[i], zs[j]]))
            
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        frames = []
        for t in range(T):
            update_frame(t)
            fig.canvas.draw()
            s, (width, height) = fig.canvas.print_to_buffer()
            image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            image = image[:, :, :3]
            frames.append(image)
        plt.close(fig)
        
        import imageio
        imageio.mimsave(save_path, frames, fps=fps)
    else:
        for t in range(T):
            update_frame(t)
            plt.pause(pause)
        plt.show()

def plot_loss_curve(train_epochs, train_losses, val_epochs, val_losses, save_path, val_smooths=None):
    if val_smooths is not None and len(val_smooths) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss subplot
        if train_losses:
            ax1.plot(train_epochs[:len(train_losses)], train_losses, label='Train Loss', color='blue', alpha=0.6)
        if val_losses:
            ax1.plot(val_epochs[:len(val_losses)], val_losses, label='Val Loss', color='orange', marker='o')
        ax1.set_xlabel('Epoch/Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Smoothness subplot
        smooth_epochs = val_epochs[:len(val_smooths)]
        ax2.plot(smooth_epochs, val_smooths, label='Val Smoothness (exp(-10*acc))', color='magenta', marker='s')
        ax2.set_xlabel('Epoch/Step')
        ax2.set_ylabel('Smoothness Score')
        ax2.set_title('Validation Action Smoothness')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_epochs[:len(train_losses)], train_losses, label='Train Loss', color='blue', alpha=0.6)
        if val_losses:
            plt.plot(val_epochs[:len(val_losses)], val_losses, label='Val Loss', color='orange', marker='o')
        plt.xlabel('Epoch/Step')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_robustness_curve(val_epochs, r_shuf_list, r_zero_list, save_path):
    plt.figure(figsize=(10, 6))
    if r_shuf_list:
        plt.plot(val_epochs[:len(r_shuf_list)], r_shuf_list, label='Robustness (Shuffle)', color='green', marker='x')
    if r_zero_list:
        plt.plot(val_epochs[:len(r_zero_list)], r_zero_list, label='Robustness (Zero)', color='red', marker='s')
        
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0)')
    plt.xlabel('Epoch/Step')
    plt.ylabel('Ratio (Error / Main Error)')
    plt.title('Conditioning Robustness (Higher is Better)')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_grpo_batch_metrics(save_dir):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    csv_path = os.path.join(save_dir, "csv_logs", "metrics.csv")
    if not os.path.exists(csv_path):
        return
        
    try:
        df = pd.read_csv(csv_path)
        df_train = df.dropna(subset=['train_r_total']).copy()
        if len(df_train) == 0:
            return
            
        x = df_train['step'].tolist()
        
        # ========== Figure 1: Training Overview (Total Reward & KL) ==========
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.plot(x, df_train['train_r_total'], 'b-', label='Total Reward (Avg)', linewidth=2)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Reward Value', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        plt.title('Training Progress: Total Reward & KL Divergence', fontsize=14, fontweight='bold')
        
        ax2 = ax1.twinx()
        ax2.plot(x, df_train['train_kl'], 'r--', label='KL Divergence', linewidth=1.5)
        ax2.set_ylabel('KL Divergence', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        loss_dir = os.path.join(save_dir, "loss")
        os.makedirs(loss_dir, exist_ok=True)
        
        loss_plot_path = os.path.join(loss_dir, 'batch_metrics_loss.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # ========== Figure 2: Rewards ==========
        fig2, ax3 = plt.subplots(figsize=(12, 6))
        
        if 'train_r_gt' in df_train.columns:
            ax3.plot(x, df_train['train_r_gt'], 'o-', label='R_GT (Ground Truth)', linewidth=2, markersize=4)
        if 'train_r_smooth' in df_train.columns:
            ax3.plot(x, df_train['train_r_smooth'], 's-', label='R_Smooth (Smoothness)', linewidth=2, markersize=4)
        if 'train_r_score' in df_train.columns:
            ax3.plot(x, df_train['train_r_score'], '^-', label='R_Score (FS Score)', linewidth=2, markersize=4)
            
        ax3.set_xlabel('Training Steps', fontsize=12)
        ax3.set_ylabel('Reward Value', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        
        plt.title('Training Progress: Rewards', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        reward_plot_path = os.path.join(loss_dir, 'batch_metrics_rewards.png')
        plt.savefig(reward_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # ========== Figure 3: Critic Value Errors & Loss (DDPO) ==========
        if 'train_critic_loss' in df_train.columns or 'train_v_err_early_t35_50' in df_train.columns:
            fig3, ax4 = plt.subplots(figsize=(12, 6))
            
            if 'train_critic_loss' in df_train.columns:
                ax4.plot(x, df_train['train_critic_loss'], 'k-', label='Critic MSE Loss', linewidth=2)
            if 'train_v_err_early_t35_50' in df_train.columns:
                ax4.plot(x, df_train['train_v_err_early_t35_50'], 'r--', label='Critic Err Early (t >= 35)', linewidth=1.8)
            if 'train_v_err_mid_t15_35' in df_train.columns:
                ax4.plot(x, df_train['train_v_err_mid_t15_35'], 'g-.', label='Critic Err Mid (15 <= t < 35)', linewidth=1.8)
            if 'train_v_err_late_t1_15' in df_train.columns:
                ax4.plot(x, df_train['train_v_err_late_t1_15'], 'b:', label='Critic Err Late (t < 15)', linewidth=1.8)
                
            ax4.set_xlabel('Training Steps', fontsize=12)
            ax4.set_ylabel('Value Error / Loss', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best', fontsize=10)
            
            plt.title('DDPO Critic Network Performance & Timestep Value Errors', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            critic_plot_path = os.path.join(loss_dir, 'batch_metrics_critic.png')
            plt.savefig(critic_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            
    except Exception as e:
        print(f"[Warning] Failed to plot GRPO batch metrics: {e}")


# ----------------- PyTorch Lightning Callback ----------------- #
class VisualizationCallback(pl.Callback):
    def __init__(self, val_interval, edges, sample_idx=0):
        super().__init__()
        self.val_interval = val_interval
        self.edges = edges
        self.sample_idx = sample_idx
        
        self.train_losses = []
        self.train_epochs = []
        self.val_losses = []
        self.val_epochs = []
        self.r_shufs = []
        self.r_zeros = []
        self.val_smooths = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        val_check = getattr(trainer, "val_check_interval", None)
        if isinstance(val_check, int) and val_check > 0:
            if trainer.global_step % 10 == 0:
                train_loss = trainer.callback_metrics.get('train_loss')
                if train_loss is not None:
                    self.train_losses.append(train_loss.item())
                    self.train_epochs.append(trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        val_check = getattr(trainer, "val_check_interval", None)
        if isinstance(val_check, int) and val_check > 0:
            return
        train_loss = trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        else:
            self.train_losses.append(0.0)
        self.train_epochs.append(trainer.current_epoch + 1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics.get('val_loss')
        val_r_shuf = trainer.callback_metrics.get('val_r_shuf')
        val_r_zero = trainer.callback_metrics.get('val_r_zero')
        
        self.val_losses.append(val_loss.item() if val_loss is not None else 0.0)
        self.r_shufs.append(val_r_shuf.item() if val_r_shuf is not None else 1.0)
        self.r_zeros.append(val_r_zero.item() if val_r_zero is not None else 1.0)
        
        val_check = getattr(trainer, "val_check_interval", None)
        if isinstance(val_check, int) and val_check > 0:
            self.val_epochs.append(trainer.global_step)
        else:
            self.val_epochs.append(epoch + 1)
        
        # 1. Compute fuller quantitative & smoothness metrics on the first validation batch
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is not None:
            if isinstance(val_dataloader, list) or isinstance(val_dataloader, tuple):
                val_dataloader = val_dataloader[0]
            try:
                # Use iter/next here to fetch the first batch of the validation set
                val_batch = next(iter(val_dataloader))
                val_batch_device = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in val_batch.items()}
                
                val_text_emb = pl_module.text_encoder(val_batch_device["motion_name"])
                
                pl_module.eval()
                with torch.no_grad():
                    val_samples, val_gt = pl_module.model.evaluate(
                        val_batch_device, n_samples=1, text_embedding=val_text_emb, sample=True
                    )[:2]
                    
                if val_gt.dim() == 3:
                    val_gt = val_gt.unsqueeze(1)
                val_gt = val_gt.expand_as(val_samples)
                
                val_input_n = pl_module.config.dataset.get("input_n", 30)
                val_p_abs = val_samples[..., val_input_n:]
                val_g_abs = val_gt[..., val_input_n:]
                
                # Compute A-MPJPE & F-MPJPE
                from utils.metrics import ampjpe, fmpjpe, compute_ldlj, compute_sparc
                
                epoch_ampjpe = float(ampjpe(val_p_abs, val_g_abs, val_p_abs.shape[2]).item())
                epoch_fmpjpe = float(fmpjpe(val_p_abs, val_g_abs, val_p_abs.shape[2]).item())
                
                # Compute LDLJ & SPARC
                pred_abs_np = val_p_abs[:, 0].transpose(-1, -2).cpu().numpy()
                ldlj_sum, sparc_sum = 0.0, 0.0
                fps = pl_module.config.dataset.get("fps", 30)
                for b in range(pred_abs_np.shape[0]):
                    ldlj_sum += compute_ldlj(pred_abs_np[b], fps=fps)
                    sparc_sum += compute_sparc(pred_abs_np[b], fps=fps)
                
                epoch_ldlj = ldlj_sum / pred_abs_np.shape[0]
                epoch_sparc = sparc_sum / pred_abs_np.shape[0]
                
                pl_module.log("val_ampjpe", epoch_ampjpe, on_step=False, on_epoch=True, sync_dist=True)
                pl_module.log("val_fmpjpe", epoch_fmpjpe, on_step=False, on_epoch=True, sync_dist=True)
                pl_module.log("val_ldlj", epoch_ldlj, on_step=False, on_epoch=True, sync_dist=True)
                pl_module.log("val_sparc", epoch_sparc, on_step=False, on_epoch=True, sync_dist=True)
            except Exception as e:
                print(f"[Warning] Failed to calculate val metrics on validation batch: {e}")
        
        # Draw a sample to evaluate and compute smoothness
        motion_path = "/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk"
        input_n = pl_module.config.dataset.get("input_n", 30)
        has_sample = False
        single_batch = None
        
        # Only use the hardcoded FineFS sample when target_dim matches (24 joints × 3 = 72)
        model_target_dim = getattr(pl_module.model, "target_dim", None)
        if model_target_dim == 24 * 3 and os.path.exists(motion_path):
            try:
                import pickle
                with open(motion_path, "rb") as f:
                    data = pickle.load(f)
                key = "pred_xyz_24_struct_global" if "pred_xyz_24_struct_global" in data else "pred_xyz_24_struct"
                xyz = data[key].astype(np.float32)
                
                output_n = pl_module.config.dataset.get("output_n", 40)
                total = input_n + output_n
                
                xyz = xyz[0 : total]
                if len(xyz) < total:
                    xyz = np.concatenate([xyz, np.repeat(xyz[-1:], total - len(xyz), axis=0)], axis=0)
                
                gt_pose = xyz.reshape(-1, 24 * 3)
                gt_tensor = torch.tensor(gt_pose, device=pl_module.device).unsqueeze(0) # (1, T, 72)
                mask_tensor = torch.zeros_like(gt_tensor)
                mask_tensor[:, :input_n] = 1
                tp_tensor = torch.arange(gt_tensor.shape[1], device=pl_module.device).unsqueeze(0).float()
                
                mode = pl_module.config.dataset.get("mode", "rotation")
                if mode == "full_name":
                    motion_name = "quadruple flip solo"
                elif mode == "rotation":
                    motion_name = "quadruple"
                else:
                    motion_name = "solo"
                
                single_batch = {
                    "pose": gt_tensor,
                    "mask": mask_tensor,
                    "timepoints": tp_tensor,
                    "motion_name": [motion_name]
                }
                has_sample = True
            except Exception as e:
                pass
                
        if not has_sample:
            val_dataloader = trainer.val_dataloaders
            if val_dataloader is not None:
                try:
                    batch = next(iter(val_dataloader))
                    single_batch = {k: v[self.sample_idx:self.sample_idx+1].to(pl_module.device) if isinstance(v, torch.Tensor) 
                                    else [v[self.sample_idx]] 
                                    for k, v in batch.items()}
                    has_sample = True
                except Exception as e:
                    pass

        val_smooth_val = 1.0
        pred_pose = None
        if has_sample and single_batch is not None:
            text_emb = pl_module.text_encoder(single_batch["motion_name"])
            pl_module.eval()
            with torch.no_grad():
                samples, _, _, _ = pl_module.model.evaluate(
                    single_batch, n_samples=1, text_embedding=text_emb, sample=True
                )
            
            pred_pose = samples[0, 0].permute(1, 0) # (L, K)
            
            # Calculate smoothness score (2nd-order derivative)
            n_joints = pl_module.model.target_dim // 3
            pred_xyz = pred_pose.view(pred_pose.shape[0], n_joints, 3) # (L, J, 3)
            vel = torch.diff(pred_xyz, dim=0) # (L-1, J, 3)
            acc = torch.diff(vel, dim=0) # (L-2, J, 3)
            acc_mag_per_joint = torch.norm(acc, dim=-1) # (L-2, J)
            start_idx = max(input_n - 2, 0)
            acc_mag_future = acc_mag_per_joint[start_idx:] # (L-2-start_idx, J)
            mean_acc = acc_mag_future.mean()
            val_smooth_val = torch.exp(-10 * mean_acc).item()
            
            # Log to loss curve / tensorboard (all processes call self.log)
            pl_module.log('val_smooth_score', val_smooth_val, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log('val_acc_mean', mean_acc.item(), on_step=False, on_epoch=True, sync_dist=True)
            
        self.val_smooths.append(val_smooth_val)
        
        # Plotting and video rendering run only on Rank 0
        if trainer.is_global_zero:
            loss_plot_path = os.path.join(trainer.logger.save_dir, "loss_curve.png")
            plot_loss_curve(self.train_epochs, self.train_losses, self.val_epochs, self.val_losses, loss_plot_path, val_smooths=self.val_smooths)
            
            robust_plot_path = os.path.join(trainer.logger.save_dir, "robustness_curve.png")
            plot_robustness_curve(self.val_epochs, self.r_shufs, self.r_zeros, robust_plot_path)
            
            # Generate the GRPO overview & rewards loss plots
            plot_grpo_batch_metrics(trainer.logger.save_dir)
            
            if pred_pose is not None:
                video_dir = os.path.join(trainer.logger.save_dir, "val_predictions")
                os.makedirs(video_dir, exist_ok=True)
                if isinstance(getattr(trainer, "val_check_interval", None), int):
                    video_path = os.path.join(video_dir, f"step_{trainer.global_step:06d}_pred.mp4")
                else:
                    video_path = os.path.join(video_dir, f"epoch_{(epoch + 1):04d}_pred.mp4")
                # Determine coordinate system orientation and playback FPS based on dataset: H36M = Y-up, Boxing = 60 FPS, FineFS/SMPL = 30 FPS
                ds_name = pl_module.config.dataset.get("name", "").lower()
                vis_y_up = True if ds_name == "h36m" else False
                vis_fps = pl_module.config.dataset.get("fps", 60 if ds_name == "boxing" else 30)
                plot_3d_motion(pred_pose, self.edges, save_path=video_path, fps=vis_fps, y_up=vis_y_up)

        # Sync all ranks to prevent NCCL timeouts while Rank 0 renders the 3D motion video
        trainer.strategy.barrier()
