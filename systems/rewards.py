import torch
import torch.nn as nn
from utils.fs_reward_utils.fs_reward_model import FSRewardModel

class UnifiedRewardModel(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.device = device
        
        if config is not None and 'rl' in config:
            self.w_gt = float(config['rl'].get('w_gt', 1.0))
            self.w_smooth = float(config['rl'].get('w_smooth', 0.5))
            self.w_score = float(config['rl'].get('w_score', 0.0))
            self.fs_ckpt = config['rl'].get('fs_reward_ckpt', None)
            
            # Read input_n from the dataset config; default to 0 if absent
            self.input_n = int(config.get('dataset', {}).get('input_n', 0))
            self.smooth_floor = float(config['rl'].get('smooth_floor', 0.0))
            self.smooth_floor_penalty = float(config['rl'].get('smooth_floor_penalty', 0.0))
            print(
                f"🔧 Reward Weights: GT={self.w_gt}, Smooth={self.w_smooth}, "
                f"Score={self.w_score}, "
                f"SmoothFloor={self.smooth_floor}, SmoothPenalty={self.smooth_floor_penalty}"
            )
        else:
            self.w_gt = 1.0
            self.w_smooth = 0.5
            self.w_score = 0.0
            self.fs_ckpt = None
            self.input_n = 0
            self.smooth_floor = 0.0
            self.smooth_floor_penalty = 0.0

        self.fs_model = None
        if self.w_score > 0 and self.fs_ckpt:
            print(f"⚖️ Loading Score Model from: {self.fs_ckpt}")
            ds_name = config.get('dataset', {}).get('name', 'finefs').lower() if config else 'finefs'
            layout = 'Body25' if ds_name == 'boxing' else 'SMPL_24'
            
            if ds_name == 'boxing':
                from utils.boxing_reward_utils.fs_reward_model import FSRewardModel as BoxingRewardModel
                self.fs_model = BoxingRewardModel(
                    checkpoint_path=self.fs_ckpt,
                    device=device,
                    layout=layout,
                    scale_type='linear',
                    min_score=-2.0,
                    max_score=2.0
                )
            else:
                self.fs_model = FSRewardModel(
                    checkpoint_path=self.fs_ckpt,
                    device=device,
                    layout=layout,
                    scale_type='linear',
                    min_score=-5.0,
                    max_score=5.0
                )
            self.fs_model.eval()
            for p in self.fs_model.parameters():

                p.requires_grad = False

    def _future_joint_mask(self, mask, samples):
        """Return (B, 1, T, J, 1) mask for future/missing joints."""
        B, _G, T, J, D = samples.shape
        if mask is None:
            future = torch.ones(B, 1, T, J, 1, device=samples.device, dtype=samples.dtype)
            if self.input_n > 0:
                future[:, :, :self.input_n] = 0.0
            return future

        mask = mask.to(samples.device).float()
        if mask.dim() != 3:
            raise ValueError(f"Expected mask with shape (B,T,K) or (B,K,T), got {mask.shape}")

        if mask.shape[1] == T:
            mask_btjd = mask.contiguous().view(B, T, J, D)
        elif mask.shape[2] == T:
            mask_btjd = mask.permute(0, 2, 1).contiguous().view(B, T, J, D)
        else:
            raise ValueError(f"Mask temporal dimension does not match samples: mask={mask.shape}, T={T}")

        observed_joint = (mask_btjd.mean(dim=-1, keepdim=True) > 0.5).float()
        return (1.0 - observed_joint).unsqueeze(1)

    def compute_gt_reward(self, samples, gt, mask=None):
        """L2 Distance Reward (0~1)"""
        diff = samples - gt
        dist_per_joint = torch.norm(diff, dim=-1)
        future_mask = self._future_joint_mask(mask, samples).squeeze(-1)
        denom = future_mask.sum(dim=(-1, -2)).clamp(min=1.0)
        dist = (dist_per_joint * future_mask).sum(dim=(-1, -2)) / denom
        reward_gt = torch.exp(-1.0 * dist)
        return reward_gt

    def compute_smoothness_reward(self, samples, mask=None, return_acc=False):
        """Boundary + future acceleration reward (0~1)."""
        if samples.shape[2] < 3:
            reward = torch.ones(samples.shape[0], samples.shape[1], device=samples.device)
            acc_mag = torch.zeros_like(reward)
            return (reward, acc_mag) if return_acc else reward

        vel = torch.diff(samples, dim=2)
        acc = torch.diff(vel, dim=2)
        acc_mag_per_joint = torch.norm(acc, dim=-1)

        future_mask = self._future_joint_mask(mask, samples).squeeze(-1)[:, :, 2:, :]
        if mask is None and self.input_n > 0:
            start = max(self.input_n - 2, 0)
            future_mask = torch.zeros_like(acc_mag_per_joint[:, :1])
            future_mask[:, :, start:, :] = 1.0

        denom = future_mask.sum(dim=(-1, -2)).clamp(min=1.0)
        acc_mag = (acc_mag_per_joint * future_mask).sum(dim=(-1, -2)) / denom
        reward_smooth = torch.exp(-10 * acc_mag)
        return (reward_smooth, acc_mag) if return_acc else reward_smooth

    def compute_score_reward(self, samples):
        """Compute the motion score"""
        if self.fs_model is None:
            return torch.zeros(samples.shape[0], samples.shape[1], device=self.device)

        B, G, T, J, D = samples.shape
        samples_flat = samples.reshape(B * G, T, J, D) 
        
        with torch.no_grad():
            scores_flat = self.fs_model(samples_flat) 
            
        scores = scores_flat.reshape(B, G)
        return scores

    def forward(self, samples, gt_motion, text_emb=None, mask=None):
        B, G, T, J, D = samples.shape
        
        if gt_motion.dim() == 4:
            gt_expanded = gt_motion.unsqueeze(1)
        else:
            gt_expanded = gt_motion.view(B, 1, T, J, D)

        r_gt = self.compute_gt_reward(samples, gt_expanded, mask=mask)
        r_smooth, acc_mag = self.compute_smoothness_reward(samples, mask=mask, return_acc=True)
        r_score = self.compute_score_reward(samples) if self.w_score > 0 else torch.zeros_like(r_gt)

        smooth_floor_penalty = torch.relu(self.smooth_floor - r_smooth) * self.smooth_floor_penalty
        
        total_reward = (
            self.w_gt * r_gt + 
            self.w_smooth * r_smooth +
            self.w_score * r_score -
            smooth_floor_penalty
        )
        
        metrics = {
            "r_gt": r_gt.mean().item(),
            "r_smooth": r_smooth.mean().item(),
            "r_score": r_score.mean().item(),
            "weighted_gt": (self.w_gt * r_gt).mean().item(),
            "weighted_smooth": (self.w_smooth * r_smooth).mean().item(),
            "acc_mean": acc_mag.mean().item(),
            "smooth_floor_penalty": smooth_floor_penalty.mean().item(),
            "total": total_reward.mean().item()
        }
        
        return total_reward, metrics
