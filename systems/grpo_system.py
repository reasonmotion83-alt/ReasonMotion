import copy
import random
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from omegaconf import OmegaConf

from modules.model import ModelMain
from utils.text_encoder import TextEncoder
from systems.rewards import UnifiedRewardModel

class GRPOSystem(pl.LightningModule):
    def __init__(self, config, target_dim: int = 72, val_dataset = None):
        super().__init__()
        self.save_hyperparameters(ignore=['config', 'val_dataset'])
        self.save_hyperparameters(config)
        self.config = config
        self.val_dataset = val_dataset
        
        # PPO / RL configurations
        rl_cfg = config.get("rl", {})
        self.ppo_epochs = rl_cfg.get("ppo_epochs", 2)
        self.epsilon = rl_cfg.get("epsilon", 0.2)
        self.kl_coef = rl_cfg.get("kl_coef", 0.05)
        self.max_kl_penalty = rl_cfg.get("kl_clip_value", None)
        self.num_samples_per_prompt = rl_cfg.get("num_samples_per_prompt", 16)
        self.sampling_std = rl_cfg.get("sampling_std", 0.02)

        # Ablation study configurations
        self.ablation_cfg = rl_cfg.get("ablation", {})
        anneal_cfg = self.ablation_cfg.get("noise_annealing", {})
        if anneal_cfg.get("enabled", False):
            self.sampling_std = anneal_cfg.get("start_std", self.sampling_std)
        
        # Delayed Sub-Trajectory settings
        self.delayed_cfg = rl_cfg.get("delayed_sub_trajectory", {})
        self.delayed_enabled = self.delayed_cfg.get("enabled", False)
        self.sampling_mode = self.delayed_cfg.get("sampling_mode", "continuous")
        self.window_size = self.delayed_cfg.get("window_size", 5)
        self.num_discrete_steps = self.delayed_cfg.get("num_discrete_steps", 5)
        
        # Reference Update
        self.update_ref_every_epoch = rl_cfg.get("update_ref_every_epoch", None)
        self.ref_ema_alpha = rl_cfg.get("ref_ema_alpha", 0.01)

        # Disable automatic optimization for PPO multiple loops
        self.automatic_optimization = False

        cfg_dict = OmegaConf.to_container(config, resolve=True)
        
        # 根據 gen_type 或 model.name 切換底層生成架構
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

        # Ensure model, old_model, and ref_model use the correct sampling_std from RL config
        self.model.sampling_std = self.sampling_std
        self.old_model.sampling_std = self.sampling_std
        self.ref_model.sampling_std = self.sampling_std

        # Reward & Text Models
        self.reward_model = UnifiedRewardModel(config=cfg_dict, device=torch.device('cpu'))
        self.text_encoder = TextEncoder(device=torch.device('cpu'))
        self.text_encoder.eval()
        for param in self.text_encoder.parameters(): param.requires_grad = False

    def on_fit_start(self):
        # Sync ref_model and old_model weights initially
        self.ref_model.load_state_dict(self.model.state_dict())
        self.old_model.load_state_dict(self.model.state_dict())
        
        # 這裡的 reward_model 需要正確分配到對應的 GPU
        self.reward_model.device = self.device
        if hasattr(self.reward_model, 'fs_model') and self.reward_model.fs_model is not None:
            self.reward_model.fs_model.to(self.device)

        # Initialize RL Visualizer
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

    def _get_active_timesteps(self):
        """決定當前 Batch 應該更新哪些時間步 (t)"""
        if not self.delayed_enabled:
            return None # None means all steps are active
            
        # ⭐️ 核心安全機制：使用當前 global_step 作為隨機種子，確保所有 Rank 抽到一模一樣的時間步！
        # 這樣就不會因為隨機狀態分歧而在 DDP backward 中發生死結。
        state = random.getstate()
        random.seed(self.global_step + 99999)
            
        num_steps = self.model.num_steps
        if self.sampling_mode == "continuous":
            t_start = random.randint(self.window_size, num_steps - 1)
            # t in (t_start - window_size, t_start]
            active = list(range(t_start - self.window_size + 1, t_start + 1))
        elif self.sampling_mode == "discrete":
            # 隨機抽取 num_discrete_steps 步
            steps = random.sample(range(1, num_steps), self.num_discrete_steps)
            active = steps
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")
            
        random.setstate(state)
        return active

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        # 1. Sync old model with current policy
        self.old_model.load_state_dict(self.model.state_dict())
        self.model.eval() # Eval mode during PPO avoids dropout randomness causing negative KL
        
        # Device assignments
        self.text_encoder.device = self.device
        texts = batch.get("motion_name", ["unknown"] * batch["pose"].shape[0])
        tok_emb, tok_mask = self.text_encoder(texts)
        text_emb = (tok_emb, tok_mask)
        
        active_timesteps = self._get_active_timesteps()
        G = self.num_samples_per_prompt
        
        with torch.no_grad():
            # 2. Rollout
            final_samples, total_log_probs, old_step_log_probs, all_latents, all_routing = self.old_model.sample_trajectory(
                text_emb, batch, G, return_step_log_probs=True, active_timesteps=active_timesteps
            )
            old_step_log_probs = old_step_log_probs.detach()
            
            # 3. Calculate Rewards
            B, _, K, L = final_samples.shape
            J = K // 3
            final_samples_reshaped = final_samples.permute(0, 1, 3, 2).reshape(B, G, L, J, 3)
            pose_gt = batch["pose"].float()
            if pose_gt.dim() == 3: pose_gt = pose_gt.view(B, L, J, 3)
            
            rewards, metrics = self.reward_model(final_samples_reshaped, pose_gt, mask=batch.get("mask"))
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
            
            clamp_cfg = self.ablation_cfg.get("advantage_clamp", {})
            if clamp_cfg.get("enabled", False):
                min_std = clamp_cfg.get("min_std", 1e-4)
                clamp_val = clamp_cfg.get("clamp_val", 3.0)
                std_r_clamped = torch.clamp(std_r, min=min_std)
                advantages = (rewards - mean_r) / std_r_clamped
                advantages = torch.clamp(advantages, min=-clamp_val, max=clamp_val).detach()
            else:
                advantages = ((rewards - mean_r) / (std_r + 1e-8)).detach()
            
            # 4. Calculate KL reference log probs
            ref_step_log_probs = self.ref_model.get_trajectory_step_log_probs(
                all_latents, text_emb, batch, all_routing=all_routing, active_timesteps=active_timesteps
            ).detach()
            
            # Approx KL stat
            log_ref_policy = torch.clamp(ref_step_log_probs - old_step_log_probs, min=-20.0, max=20.0)
            approx_kl = torch.exp(log_ref_policy) - log_ref_policy - 1.0
            kl_div = approx_kl.mean().item()

        # 5. PPO Updates
        self.model.eval() # 保持 eval 模式以避免去噪過程中的 dropout 導致機率估計不一致，從而穩定 KL 與梯度
        for ppo_epoch in range(self.ppo_epochs):
            opt.zero_grad()
            self.model.backprop_trajectory_loss(
                all_latents,
                text_emb,
                batch,
                advantages,
                old_step_log_probs=old_step_log_probs,
                ref_step_log_probs=ref_step_log_probs,
                epsilon=self.epsilon,
                kl_coef=self.kl_coef,
                max_kl_penalty=self.max_kl_penalty,
                all_routing=all_routing,
                active_timesteps=active_timesteps,
                backward_fn=self.manual_backward,
                ablation_cfg=self.ablation_cfg,
                pl_module=self
            )
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()
            
        lr_scheduler = self.lr_schedulers()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Adaptive KL Penalty Coefficient Update
        adaptive_kl_cfg = self.ablation_cfg.get("adaptive_kl", {})
        if adaptive_kl_cfg.get("enabled", False):
            target_kl = adaptive_kl_cfg.get("target_kl", 0.05)
            kl_min = adaptive_kl_cfg.get("kl_min", 0.001)
            kl_max = adaptive_kl_cfg.get("kl_max", 0.5)
            if kl_div > target_kl * 1.5:
                self.kl_coef = min(kl_max, self.kl_coef * 1.5)
            elif kl_div < target_kl / 1.5:
                self.kl_coef = max(kl_min, self.kl_coef / 1.5)
            self.log('train_kl_coef', self.kl_coef, sync_dist=True)

        # Logging
        self.log('train_loss', -rewards.mean(), sync_dist=True)
        self.log('train_r_total', rewards.mean(), prog_bar=True, sync_dist=True)
        self.log('train_kl', kl_div, prog_bar=True, sync_dist=True)
        self.log('train_r_gt', metrics['r_gt'], sync_dist=True)
        self.log('train_r_smooth', metrics['r_smooth'], sync_dist=True)
        if self.reward_model.w_score > 0:
            self.log('train_r_score', metrics['r_score'], prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self):
        # KL Reference EMA Update
        if self.update_ref_every_epoch and (self.current_epoch + 1) % self.update_ref_every_epoch == 0:
            alpha = self.ref_ema_alpha
            if alpha == 1.0:
                self.ref_model.load_state_dict(self.model.state_dict())
            elif alpha > 0.0:
                with torch.no_grad():
                    # Handle DataParallel/DistributedDataParallel if wrapped, but here we assume simple nn.Module
                    for ref_param, param in zip(self.ref_model.parameters(), self.model.parameters()):
                        ref_param.data.mul_(1.0 - alpha).add_(param.data, alpha=alpha)

        # Noise Annealing Update
        anneal_cfg = self.ablation_cfg.get("noise_annealing", {})
        if anneal_cfg.get("enabled", False):
            min_std = anneal_cfg.get("min_std", 0.005)
            decay = anneal_cfg.get("decay_rate", 0.95)
            self.sampling_std = max(min_std, self.sampling_std * decay)
            # 同步更新底層 model 的 sampling_std
            self.model.sampling_std = self.sampling_std
            self.old_model.sampling_std = self.sampling_std
            self.ref_model.sampling_std = self.sampling_std
            self.log('sampling_std', self.sampling_std, sync_dist=True)

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
            
        # Logging val_loss as negative total reward (so lower is better for ModelCheckpoint)
        val_loss = -rewards.mean()
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.log('val_r_total', rewards.mean(), sync_dist=True)
        self.log('val_r_score', metrics['r_score'], sync_dist=True)
        self.log('epoch_1based', float(self.current_epoch + 1), on_step=False, on_epoch=True, sync_dist=True)
        
        # Run RL-specific visualization
        if batch_idx == 0 and hasattr(self, "visualizer") and self.trainer.is_global_zero:
            self.visualizer.run_epoch_viz(
                epoch=self.current_epoch,
                model=self.model,
                reward_model=self.reward_model,
                text_encoder=self.text_encoder,
                current_std=self.sampling_std,
                num_variants=10, # Generate 10 variants for visualization
                config=self.config,
                step=self.trainer.global_step
            )
            
        # Sync all ranks to prevent NCCL timeouts while Rank 0 renders the video
        self.trainer.strategy.barrier()
            
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.config.training.lr)
        
        sched_conf = self.config.rl.get('lr_schedule', {})
        sched_type = sched_conf.get('type', 'constant')
        
        if sched_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched_conf.get('step_size', 5000), gamma=sched_conf.get('gamma', 0.5)
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sched_conf.get('t_max', 30000), eta_min=sched_conf.get('eta_min', 1e-7)
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
            
        return optimizer

    def train(self, mode: bool = True):
        super().train(mode)
        # 強制這些靜態/凍結/參考子模型永遠保持在 eval 模式
        if hasattr(self, "reward_model") and self.reward_model is not None:
            self.reward_model.eval()
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.eval()
        if hasattr(self, "ref_model") and self.ref_model is not None:
            self.ref_model.eval()
        if hasattr(self, "old_model") and self.old_model is not None:
            self.old_model.eval()
