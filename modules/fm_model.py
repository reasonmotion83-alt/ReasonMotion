import torch
import numpy as np
import math
from modules.model import ModelMain

class FlowMatchingModel(ModelMain):
    """
    Experimental Flow Matching (Rectified Flow) Model.
    繼承現有網路架構，將原本的預測雜訊 (Diffusion) 替換為預測向量場 (Vector Field)。
    完全不改變底層的 Transformer 參數維度，並擴充支援連續時間 GRPO 強化學習訓練。
    """
    def __init__(self, config, device, target_dim):
        super().__init__(config, device, target_dim)
        print("🌊 [Experimental] Flow Matching Model Activated!")

    def forward(self, batch, is_train=True, text_embedding=None):
        pose, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        
        B, K, L = pose.shape
        device = pose.device
        
        # 1. 抽樣連續時間 t ~ U(1e-5, 1.0)
        t = torch.rand((B,), device=device, dtype=pose.dtype) * (1.0 - 1e-5) + 1e-5
        
        # 2. 生成標準高斯雜訊 (x1)
        x1 = torch.randn_like(pose)
        
        # 3. 沿著直線插值得到當前狀態 x_t
        t_expand = t.view(B, 1, 1)
        x_t = t_expand * x1 + (1.0 - t_expand) * pose
        
        # 4. 目標向量場 (Vector Field)
        v_target = x1 - pose
        
        # 5. 網路預測向量場
        # 將 t 放縮到網路習慣的尺度 (0 ~ num_steps) 以適配原有的 Sinusoidal Embedding
        t_scaled = t * self.num_steps
        
        inp = self.set_input_to_diffmodel(x_t, pose, mask)
        v_pred = self.diffmodel(inp, side_info, t_scaled, text_emb=text_embedding)
        v_pred = v_pred.reshape(B, K, L)
        
        # 6. 計算 Reconstruction Loss (只在 1 - cond_mask 的區域計算)
        loss = ((v_pred - v_target) * (1 - mask)).pow(2).sum()
        denom = (1 - mask).sum().clamp(min=1)
        loss = loss / denom
        
        # MoE load balancing loss (如果存在)
        if self.balancing_loss:
            if hasattr(self.diffmodel, 'get_load_balancing_loss'):
                load_balancing_loss, _total_fi = self.diffmodel.get_load_balancing_loss()
                loss += self.balance_loss_weight * load_balancing_loss
                
        return loss

    @torch.no_grad()
    def evaluate(self, batch, n_samples, text_embedding=None, noisy_data=None, sample=True):
        """
        Euler ODE Solver for Flow Matching
        dx/dt = v_theta(x, t)
        """
        pose, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        B, K, L = pose.shape
        device = pose.device
        
        # Repeat condition for n_samples
        side_rep = side_info.repeat_interleave(n_samples, dim=0)
        cond_rep = mask.repeat_interleave(n_samples, dim=0)
        obs_rep = pose.repeat_interleave(n_samples, dim=0)
        
        if isinstance(text_embedding, tuple):
            text_rep = tuple(u.repeat_interleave(n_samples, dim=0) for u in text_embedding)
        else:
            text_rep = text_embedding.repeat_interleave(n_samples, dim=0) if text_embedding is not None else None
            
        # 初始化 x_1 (Noise)
        if noisy_data is not None:
            if noisy_data.ndim == 4 and noisy_data.shape[1] == n_samples:
                x_t = noisy_data.to(device=device, dtype=pose.dtype).clone().reshape(B * n_samples, K, L)
            elif noisy_data.ndim == 3:
                x_t = noisy_data.to(device=device, dtype=pose.dtype).unsqueeze(1).expand(-1, n_samples, -1, -1).clone().reshape(B * n_samples, K, L)
            else:
                x_t = noisy_data.to(device=device, dtype=pose.dtype).clone()
        else:
            x_t = torch.randn(B * n_samples, K, L, device=device, dtype=pose.dtype)
            
        # Euler 積分步長
        dt = 1.0 / self.num_steps
        
        # 從 t=1 往 t=0 解 ODE (Noise -> Data)
        t_steps = torch.linspace(1.0, dt, self.num_steps, device=device)
        
        for t in t_steps:
            t_batch = torch.full((B * n_samples,), t.item() * self.num_steps, device=device, dtype=pose.dtype)
            inp = self.set_input_to_diffmodel(x_t, obs_rep, cond_rep)
            
            v_pred = self.diffmodel(inp, side_rep, t_batch, text_emb=text_rep)
            
            # Euler ODE step: x_{t-dt} = x_t - v * dt
            x_t = x_t - v_pred * dt
            
            # 在條件幀強制覆蓋真實資料 (In-painting constraint)
            x_t = (1 - cond_rep) * x_t + cond_rep * obs_rep
            
        final_samples = x_t.reshape(B, n_samples, K, L)
        return final_samples, pose, (1 - mask), tp

    # =========================================================================
    # 🌟 Flow Matching GRPO Reinforcement Learning (RL) 支援方法
    # =========================================================================

    def sample_trajectory(self, text_emb, batch, G, return_step_log_probs=False, active_timesteps=None, diffmodel_hook=None):
        """
        stochastic policy rollout based on Euler SDE:
        x_{t - dt} = x_t - v_pred * dt + sigma * z
        """
        if active_timesteps is not None:
            active_timesteps = set(active_timesteps)
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape
        device = observed_data.device

        noise_scale = max(float(getattr(self, "sampling_std", 1.0)), 1e-6)
        # 起點 t=1.0 的標準高斯雜訊
        x_t = torch.randn(B, 1, K, L, device=device).expand(-1, G, -1, -1).clone()
        all_latents = [x_t]
        total_log_probs = torch.zeros(B, G, device=device)
        step_log_probs = []
        all_routing = [] if self.use_r3 else None

        side_rep  = side_info.repeat_interleave(G, dim=0)
        cond_rep  = gt_mask.repeat_interleave(G, dim=0)
        obs_rep   = observed_data.repeat_interleave(G, dim=0)
        future_mask = (1 - gt_mask).repeat_interleave(G, dim=0)
        future_denom = future_mask.sum(dim=(1, 2)).clamp(min=1.0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0) if text_emb is not None else None

        dt = 1.0 / self.num_steps
        t_steps = torch.linspace(1.0, dt, self.num_steps, device=device)

        for step_idx, t in enumerate(t_steps):
            x_t_flat = x_t.reshape(B * G, K, L)
            inp = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            
            # 使用連續浮點數時間輸入
            t_vec = torch.full((B * G,), t.item() * self.num_steps, dtype=observed_data.dtype, device=device)
            
            if diffmodel_hook is not None:
                v_pred = diffmodel_hook(self.diffmodel, t_index, inp, side_rep, t_vec, text_rep)
            else:
                v_pred = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)
            v_pred = v_pred.reshape(B * G, K, L)

            if self.use_r3:
                all_routing.append(self._collect_moe_routing())

            mean = x_t_flat - v_pred * dt
            t_index = self.num_steps - 1 - step_idx

            if step_idx < self.num_steps - 1:
                is_active = (t_index in active_timesteps) if active_timesteps is not None else True
                if is_active:
                    # ⭐️ 數學優化：將噪聲標準差縮放 sqrt(dt)，確保 N 步隨機漫步後的總變異數收斂在 sampling_std^2，防止數值爆炸
                    sigma = noise_scale * math.sqrt(dt)
                    z = torch.randn_like(mean)
                    x_prev = mean + sigma * z
                    # 步驟對數機率 (Log-probability)
                    step_lp_elem = -0.5 * z ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
                    step_lp = (step_lp_elem * future_mask).sum(dim=(1, 2)) / future_denom
                    step_lp = step_lp.reshape(B, G)
                    total_log_probs += step_lp
                    step_log_probs.append(step_lp)
                else:
                    shared_noise_scale = float(getattr(self, "shared_noise_scale", 1.0))
                    sigma = shared_noise_scale * math.sqrt(dt)
                    z_shared = torch.randn(B, 1, K, L, device=device).expand(-1, G, -1, -1).reshape(B * G, K, L)
                    z = z_shared
                    x_prev = mean + sigma * z
                
                x_prev = (1 - cond_rep) * x_prev + cond_rep * obs_rep
                x_t = x_prev.reshape(B, G, K, L)
            else:
                # 最後一步為確定性映射到 t=0
                x_prev = mean
                x_prev = (1 - cond_rep) * x_prev + cond_rep * obs_rep
                x_t = x_prev.reshape(B, G, K, L)

            all_latents.append(x_t)
            x_t = x_t.detach()

        final_samples = x_t * (1 - gt_mask.unsqueeze(1)) + observed_data.unsqueeze(1) * gt_mask.unsqueeze(1)
        if step_log_probs:
            total_log_probs = total_log_probs / len(step_log_probs)
            step_log_probs = torch.stack(step_log_probs, dim=0)
        else:
            step_log_probs = torch.empty(0, B, G, device=device)
            
        if return_step_log_probs:
            return final_samples, total_log_probs, step_log_probs, all_latents, all_routing
        return final_samples, total_log_probs, all_latents, all_routing

    def get_trajectory_step_log_probs(self, all_latents, text_emb, batch, all_routing=None, active_timesteps=None):
        if active_timesteps is not None:
            active_timesteps = set(active_timesteps)
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape
        G = all_latents[0].shape[1]
        device = observed_data.device

        noise_scale = max(float(getattr(self, "sampling_std", 1.0)), 1e-6)
        step_log_probs = []

        side_rep  = side_info.repeat_interleave(G, dim=0)
        cond_rep  = gt_mask.repeat_interleave(G, dim=0)
        obs_rep   = observed_data.repeat_interleave(G, dim=0)
        future_mask = (1 - gt_mask).repeat_interleave(G, dim=0)
        future_denom = future_mask.sum(dim=(1, 2)).clamp(min=1.0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0) if text_emb is not None else None

        dt = 1.0 / self.num_steps
        t_steps = torch.linspace(1.0, dt, self.num_steps, device=device)

        for step_idx, t in enumerate(t_steps):
            t_index = self.num_steps - 1 - step_idx
            if step_idx >= self.num_steps - 1:
                continue

            is_active = (t_index in active_timesteps) if active_timesteps is not None else True
            if not is_active:
                continue

            x_t = all_latents[step_idx]
            x_prev = all_latents[step_idx + 1]

            x_t_flat = x_t.reshape(B * G, K, L)
            inp = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t.item() * self.num_steps, dtype=observed_data.dtype, device=device)

            if self.use_r3 and all_routing is not None:
                self._restore_moe_routing(all_routing[step_idx])

            v_pred = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)
            v_pred = v_pred.reshape(B * G, K, L)

            mean = x_t_flat - v_pred * dt
            x_prev_flat = x_prev.reshape(B * G, K, L)

            # ⭐️ 數學優化：對齊縮放後的噪聲標準差 sigma
            sigma = noise_scale * math.sqrt(dt)
            z = (x_prev_flat - mean) / sigma
            step_lp_elem = -0.5 * z ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
            step_lp = (step_lp_elem * future_mask).sum(dim=(1, 2)) / future_denom
            step_lp = step_lp.reshape(B, G)
            step_log_probs.append(step_lp)

        if step_log_probs:
            return torch.stack(step_log_probs, dim=0)
        return torch.empty(0, B, G, device=device)

    def backprop_trajectory_loss(self, all_latents, text_emb, batch, advantages, old_step_log_probs, ref_step_log_probs, epsilon, kl_coef, max_kl_penalty, all_routing=None, active_timesteps=None, backward_fn=None, ablation_cfg=None, pl_module=None):
        # Check if DDP is active and retrieve the no_sync context manager
        is_ddp = False
        no_sync_ctx = None
        if pl_module is not None and getattr(pl_module, "_trainer", None) is not None:
            strategy = pl_module._trainer.strategy
            if hasattr(strategy, "model") and strategy.model is not None:
                if hasattr(strategy.model, "no_sync"):
                    is_ddp = True
                    no_sync_ctx = strategy.model.no_sync

        if active_timesteps is not None:
            active_timesteps = set(active_timesteps)
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape
        G = all_latents[0].shape[1]
        device = observed_data.device

        noise_scale = max(float(getattr(self, "sampling_std", 1.0)), 1e-6)

        side_rep  = side_info.repeat_interleave(G, dim=0)
        cond_rep  = gt_mask.repeat_interleave(G, dim=0)
        obs_rep   = observed_data.repeat_interleave(G, dim=0)
        future_mask = (1 - gt_mask).repeat_interleave(G, dim=0)
        future_denom = future_mask.sum(dim=(1, 2)).clamp(min=1.0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0) if text_emb is not None else None

        dt = 1.0 / self.num_steps
        t_steps = torch.linspace(1.0, dt, self.num_steps, device=device)

        ablation = ablation_cfg if ablation_cfg is not None else {}
        decay_adv_cfg = ablation.get("step_decay_advantage", {})
        topk_gating_cfg = ablation.get("topk_gating", {})

        active_steps = len(active_timesteps) if active_timesteps is not None else (self.num_steps - 1)
        active_step_idx = 0

        for step_idx, t in enumerate(t_steps):
            t_index = self.num_steps - 1 - step_idx
            if step_idx >= self.num_steps - 1:
                continue

            is_active = (t_index in active_timesteps) if active_timesteps is not None else True
            if not is_active:
                continue

            active_idx = active_step_idx

            x_t = all_latents[step_idx]
            x_prev = all_latents[step_idx + 1]

            x_t_flat = x_t.reshape(B * G, K, L).detach()
            inp = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t.item() * self.num_steps, dtype=observed_data.dtype, device=device)

            if self.use_r3 and all_routing is not None:
                self._restore_moe_routing(all_routing[step_idx])

            v_pred = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)
            v_pred = v_pred.reshape(B * G, K, L)

            mean = x_t_flat - v_pred * dt
            x_prev_flat = x_prev.reshape(B * G, K, L).detach()

            # ⭐️ 數學優化：對齊縮放後的噪聲標準差 sigma
            sigma = noise_scale * math.sqrt(dt)
            z = (x_prev_flat - mean) / sigma
            step_lp_elem = -0.5 * z ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
            step_lp = (step_lp_elem * future_mask).sum(dim=(1, 2)) / future_denom
            step_lp = step_lp.reshape(B, G)

            old_lp = old_step_log_probs[active_idx].to(device).detach()
            ref_lp = ref_step_log_probs[active_idx].to(device).detach()

            ratio = torch.exp(torch.clamp(step_lp - old_lp, min=-20.0, max=20.0))
            
            adv_scale = advantages
            if decay_adv_cfg.get("enabled", False):
                decay_power = decay_adv_cfg.get("power", 1.0)
                step_decay = (t.item()) ** decay_power
                adv_scale = advantages * step_decay

            surr1 = ratio * adv_scale
            surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv_scale
            policy_loss = -torch.min(surr1, surr2)

            log_ref_policy = torch.clamp(ref_lp - step_lp, min=-20.0, max=20.0)
            kl_penalty = torch.exp(log_ref_policy) - log_ref_policy - 1.0
            if max_kl_penalty is not None:
                kl_penalty = torch.clamp(kl_penalty, max=max_kl_penalty)

            step_loss = policy_loss + kl_coef * kl_penalty

            if topk_gating_cfg.get("enabled", False):
                topk_ratio = topk_gating_cfg.get("ratio", 0.5)
                k_val = max(1, int(G * topk_ratio))
                threshold = torch.topk(advantages, k_val, dim=1).values[:, -1:]
                gating_mask = (advantages >= threshold).float()
                step_loss = step_loss * gating_mask

            loss_step = (step_loss.mean() / active_steps)

            # Check if this is the last active step in the loop
            is_last_step = (active_step_idx == active_steps - 1)

            # Backward step-by-step to save memory
            if backward_fn is not None:
                if is_ddp and not is_last_step:
                    with no_sync_ctx():
                        backward_fn(loss_step)
                else:
                    backward_fn(loss_step)
            else:
                if is_ddp and not is_last_step:
                    with no_sync_ctx():
                        loss_step.backward()
                else:
                    loss_step.backward()

            # Free up VRAM immediately
            del loss_step, step_loss, step_lp, v_pred, mean, inp
            active_step_idx += 1
