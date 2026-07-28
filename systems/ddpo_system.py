import copy
import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from omegaconf import OmegaConf

from modules.model import ModelMain
from utils.text_encoder import TextEncoder
from systems.rewards import UnifiedRewardModel

class DDPOSystem(pl.LightningModule):
    """
    Strict DDPO (Denoising Diffusion Policy Optimization, Black et al., ICLR 2024)
    system implementation for 3D Motion Diffusion.
    
    Key Features matching Black et al. (2023):
    1. Critic-Free: NO neural network Critic (V_phi).
    2. Per-Sample / Per-Condition Running Baseline: Tracks running mean & std across epochs
       keyed by each dataset sample's unique ID (or idx), preserving identical 30-frame prefix condition c_i.
    3. Small Clipping Range: Default epsilon = 1e-4 (0.0001).
    4. Single Optimizer: Only optimizes policy UNet parameters.
    """
    def __init__(self, config, target_dim: int = 72, val_dataset = None):
        super().__init__()
        self.save_hyperparameters(ignore=['config', 'val_dataset'])
        self.save_hyperparameters(config)
        self.config = config
        self.val_dataset = val_dataset
        
        # Strict DDPO RL configurations
        rl_cfg = config.get("rl", {})
        self.ppo_epochs = rl_cfg.get("ppo_epochs", 2)
        # DDPO Paper Appendix D.1 explicitly uses clip range = 1e-4
        self.epsilon = rl_cfg.get("epsilon", 1e-4)
        self.kl_coef = rl_cfg.get("kl_coef", 0.05)
        self.max_kl_penalty = rl_cfg.get("kl_clip_value", None)
        self.num_samples_per_prompt = rl_cfg.get("num_samples_per_prompt", 16)
        self.sampling_std = rl_cfg.get("sampling_std", 1.0)

        # Disable Lightning automatic optimization for manual PPO optimization loop
        self.automatic_optimization = False

        cfg_dict = OmegaConf.to_container(config, resolve=True)
        
        gen_type = config.get("training", {}).get("gen_type", "ddpm")
        model_name = config.get("model", {}).get("name", "sft_baseline")
        if gen_type == "flow_matching" or model_name == "flow_matching":
            from modules.fm_model import FlowMatchingModel
            model_class = FlowMatchingModel
        else:
            model_class = ModelMain
            
        # 1. Policy Model (Trainable)
        self.model = model_class(cfg_dict["model"], device=torch.device('cpu'), target_dim=target_dim)
        
        # 2. KL Reference Model (Frozen)
        self.ref_model = model_class(cfg_dict["model"], device=torch.device('cpu'), target_dim=target_dim)
        self.ref_model.eval()
        for p in self.ref_model.parameters(): p.requires_grad = False
            
        # 3. Old Policy Model for Rollout (Frozen, syncs with model each batch)
        self.old_model = model_class(cfg_dict["model"], device=torch.device('cpu'), target_dim=target_dim)
        self.old_model.eval()
        for p in self.old_model.parameters(): p.requires_grad = False

        self.model.sampling_std = self.sampling_std
        self.old_model.sampling_std = self.sampling_std
        self.ref_model.sampling_std = self.sampling_std

        # Per-Sample / Per-Condition Running Mean & Std Tracker (DDPO Paper Appendix D.3)
        # Structure: {sample_key: {"mean": float, "std": float, "count": int}}
        self.running_stats = {}

        # Reward & Text Models
        self.reward_model = UnifiedRewardModel(config=cfg_dict, device=torch.device('cpu'))
        self.text_encoder = TextEncoder(device=torch.device('cpu'))
        self.text_encoder.eval()
        for param in self.text_encoder.parameters(): param.requires_grad = False

    def on_fit_start(self):
        self.ref_model.load_state_dict(self.model.state_dict())
        self.old_model.load_state_dict(self.model.state_dict())
        
        self.reward_model.device = self.device
        if hasattr(self.reward_model, 'fs_model') and self.reward_model.fs_model is not None:
            self.reward_model.fs_model.to(self.device)

        from utils.rl_visualizer import RLVisualizer
        output_dir = getattr(self.trainer.logger, "save_dir", "outputs/rl_vis") if self.trainer.logger else "outputs/rl_vis"
        self.visualizer = RLVisualizer(
            output_dir=output_dir,
            val_idx=0,
            top_k=3, bot_k=3,
            viz_mode="all",
            device=self.device,
            joints=self.model.target_dim // 3
        )
        val_dataset = self.val_dataset
        if val_dataset is None and self.trainer.val_dataloaders is not None:
            val_dataset = self.trainer.val_dataloaders.dataset
            
        if val_dataset is not None:
            self.visualizer.load_fixed_sample(val_dataset)

    def _normalize_rewards_per_sample(self, rewards, sample_keys):
        """
        Strict DDPO Reward Normalization per Appendix D.3 adapted for motion prediction:
        Normalizes rewards by running mean and std tracked for each dataset sample independently across epochs.
        rewards: Tensor of shape (B, G)
        sample_keys: list of unique sample identifiers of length B
        """
        B, G = rewards.shape
        normalized_adv = torch.zeros_like(rewards)
        
        for b in range(B):
            key = str(sample_keys[b])
            r_batch = rewards[b] # (G,)
            r_mean_current = r_batch.mean().item()
            r_std_current = r_batch.std(unbiased=False).item()
            
            # Update running statistics for this specific dataset sample (exact same 30-frame prefix)
            if key not in self.running_stats:
                self.running_stats[key] = {
                    "mean": r_mean_current,
                    "std": max(r_std_current, 1e-4),
                    "count": G
                }
            else:
                stats = self.running_stats[key]
                old_count = stats["count"]
                new_count = old_count + G
                # Cumulative running mean and std update across epochs
                new_mean = (stats["mean"] * old_count + r_mean_current * G) / new_count
                new_std = (stats["std"] * old_count + max(r_std_current, 1e-4) * G) / new_count
                stats["mean"] = new_mean
                stats["std"] = max(new_std, 1e-4)
                stats["count"] = new_count
                
            stats = self.running_stats[key]
            adv_b = (r_batch - stats["mean"]) / (stats["std"] + 1e-8)
            normalized_adv[b] = adv_b
            
        return normalized_adv

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opt_policy = opts[0] if isinstance(opts, (list, tuple)) else opts
        
        # 1. Sync old model
        self.old_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        
        self.text_encoder.device = self.device
        texts = batch.get("motion_name", ["unknown"] * batch["pose"].shape[0])
        tok_emb, tok_mask = self.text_encoder(texts)
        text_emb = (tok_emb, tok_mask)
        
        # Determine unique sample keys for per-sample tracking across epochs
        if "sample_id" in batch:
            sample_keys = batch["sample_id"]
        elif "idx" in batch:
            sample_keys = [str(x.item()) for x in batch["idx"]]
        else:
            sample_keys = texts
        
        G = self.num_samples_per_prompt
        
        with torch.no_grad():
            # 2. Rollout across ALL timesteps (active_timesteps = None for pure DDPO)
            final_samples, total_log_probs, old_step_log_probs, all_latents, all_routing = self.old_model.sample_trajectory(
                text_emb, batch, G, return_step_log_probs=True, active_timesteps=None
            )
            old_step_log_probs = old_step_log_probs.detach()
            
            # 3. Calculate Trajectory Rewards R(x_0)
            B, _, K, L = final_samples.shape
            J = K // 3
            final_samples_reshaped = final_samples.permute(0, 1, 3, 2).reshape(B, G, L, J, 3)
            pose_gt = batch["pose"].float()
            if pose_gt.dim() == 3: pose_gt = pose_gt.view(B, L, J, 3)
            
            rewards, metrics = self.reward_model(final_samples_reshaped, pose_gt, mask=batch.get("mask")) # (B, G)

        # 4. Strict DDPO Per-Sample Advantage Calculation (Appendix D.3)
        # Compute normalized advantages for each sample's unique condition (B, G)
        prompt_adv = self._normalize_rewards_per_sample(rewards, sample_keys)
        
        # Expand across all diffusion timesteps: (num_steps - 1, B, G)
        num_steps = self.model.num_steps
        step_advantages = prompt_adv.unsqueeze(0).repeat(num_steps - 1, 1, 1)

        # 5. Calculate KL reference log probs
        with torch.no_grad():
            ref_step_log_probs = self.ref_model.get_trajectory_step_log_probs(
                all_latents, text_emb, batch, all_routing=all_routing, active_timesteps=None
            ).detach()
            
            log_ref_policy = torch.clamp(ref_step_log_probs - old_step_log_probs, min=-20.0, max=20.0)
            approx_kl = torch.exp(log_ref_policy) - log_ref_policy - 1.0
            kl_div = approx_kl.mean().item()

        # 6. Optimize Policy Network (Strict DDPO Updates with clip_range = 1e-4)
        self.model.eval()
        for ppo_epoch in range(self.ppo_epochs):
            opt_policy.zero_grad()
            self.model.backprop_trajectory_loss(
                all_latents,
                text_emb,
                batch,
                step_advantages,
                old_step_log_probs=old_step_log_probs,
                ref_step_log_probs=ref_step_log_probs,
                epsilon=self.epsilon, # 1e-4
                kl_coef=self.kl_coef,
                max_kl_penalty=self.max_kl_penalty,
                all_routing=all_routing,
                active_timesteps=None,
                backward_fn=self.manual_backward,
                ablation_cfg=None, # Pure DDPO: NO custom ablations!
                pl_module=self
            )
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt_policy.step()

        self.log('train_loss', -rewards.mean(), sync_dist=True)
        self.log('train_r_total', rewards.mean(), prog_bar=True, sync_dist=True)
        self.log('train_kl', kl_div, prog_bar=True, sync_dist=True)
        self.log('train_r_gt', metrics['r_gt'], sync_dist=True)
        self.log('train_r_smooth', metrics['r_smooth'], sync_dist=True)
        if self.reward_model.w_score > 0:
            self.log('train_r_score', metrics['r_score'], prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.text_encoder.device = self.device
        texts = batch.get("motion_name", ["unknown"] * batch["pose"].shape[0])
        tok_emb, tok_mask = self.text_encoder(texts)
        text_emb = (tok_emb, tok_mask)
        
        with torch.no_grad():
            samples, total_log_probs, all_latents, all_routing = self.model.sample_trajectory(
                text_emb, batch, G=1, return_step_log_probs=False
            )
            
            B, _, K, L = samples.shape
            J = K // 3
            samples_reshaped = samples.permute(0, 1, 3, 2).reshape(B, 1, L, J, 3)
            pose_gt = batch["pose"].float().view(B, L, J, 3)
            
            rewards, metrics = self.reward_model(samples_reshaped, pose_gt, mask=batch.get("mask"))
            
        val_loss = -rewards.mean()
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.log('val_r_total', rewards.mean(), sync_dist=True)
        self.log('val_r_score', metrics['r_score'], sync_dist=True)
        self.log('epoch_1based', float(self.current_epoch + 1), on_step=False, on_epoch=True, sync_dist=True)
        
        if batch_idx == 0 and hasattr(self, "visualizer") and self.trainer.is_global_zero:
            self.visualizer.run_epoch_viz(
                epoch=self.current_epoch,
                model=self.model,
                reward_model=self.reward_model,
                text_encoder=self.text_encoder,
                current_std=self.sampling_std,
                num_variants=10,
                config=self.config,
                step=self.trainer.global_step
            )
            
        self.trainer.strategy.barrier()
        return val_loss

    def configure_optimizers(self):
        opt_policy = Adam(self.model.parameters(), lr=self.config.training.lr)
        sched_conf = self.config.rl.get('lr_schedule', {})
        sched_type = sched_conf.get('type', 'constant')
        
        if sched_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt_policy, step_size=sched_conf.get('step_size', 5000), gamma=sched_conf.get('gamma', 0.5)
            )
            return [opt_policy], [{"scheduler": scheduler, "interval": "step"}]
            
        return [opt_policy]

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "reward_model") and self.reward_model is not None:
            self.reward_model.eval()
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.eval()
        if hasattr(self, "ref_model") and self.ref_model is not None:
            self.ref_model.eval()
        if hasattr(self, "old_model") and self.old_model is not None:
            self.old_model.eval()
