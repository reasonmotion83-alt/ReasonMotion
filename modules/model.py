import math
import torch
import torch.nn as nn
import numpy as np
from modules.diffusion_util import diff_CSDI

class ModelMain(nn.Module):
    def __init__(self, config, device, target_dim=24 * 3):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        
        # 讀取擴散模型設定
        diff_cfg = config["diffusion"]
        self.is_unconditional = bool(diff_cfg["is_unconditional"])
        self.time_step = 0
        
        # ---- Embedding: 時間位置編碼 + joint ID 編碼 ----
        self.emb_time_dim = diff_cfg["timeemb_dim"]
        self.emb_feature_dim = diff_cfg["featureemb_dim"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for cond_mask

        self.embed_layer = nn.Embedding(self.target_dim, self.emb_feature_dim)

        # ---- GRPO / RL config ----
        rl_config = config.get("rl", {})
        self.sampling_std = float(rl_config.get("sampling_std", 0.05))

        # ---- MoE 設定 ----
        moe_cfg = config.get("moe", {})
        self.balancing_loss = moe_cfg.get("balance_loss", False)
        self.balance_loss_weight = moe_cfg.get("balance_loss_weight", 0.01)
        print(f"Using load balancing loss: {self.balancing_loss}, weight: {self.balance_loss_weight}")
        
        self.use_r3 = moe_cfg.get("use_r3", False)
        if self.use_r3:
            print("[R3] Strategy C enabled — two-pass detached routing replay")

        # ---- Diffusion 主體 ----
        cfg_diff = diff_cfg.copy()
        cfg_diff["side_dim"] = self.emb_total_dim
        cfg_diff["textemb"] = config["textemb_dim"]
        cfg_diff["moe"] = moe_cfg.get("enabled", False)

        in_channels = 1 if self.is_unconditional else 2
        if moe_cfg.get("multirouter", False):
            raise NotImplementedError("Multirouter is not implemented in this codebase.")
        else:
            print("Using diff_CSDI for MoE without multirouter")
            self.diffmodel = diff_CSDI(cfg_diff, in_channels, text_mode=config.get("text_mode", "cross_attn"))
        
        # ---- beta schedule ----
        self.num_steps = cfg_diff["num_steps"]
        if "schedule" in cfg_diff:
            self.beta = self._make_beta_schedule(cfg_diff)
            self.alpha_hat = 1 - self.beta
            self.alpha = np.cumprod(self.alpha_hat)
            self.alpha_torch = torch.tensor(self.alpha, dtype=torch.float32, device=self.device).view(-1, 1, 1)
        else:
            self.beta = None
            self.alpha_hat = None
            self.alpha = None
            self.alpha_torch = None


    @property
    def current_device(self):
        return self.embed_layer.weight.device

    def forward(self, batch, is_train=True, text_embedding=None):
        pose, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        if self.use_r3 and is_train:
            return self._calc_loss_r3(pose, mask, side_info, text_emb=text_embedding)
        return self._calc_loss(pose, mask, side_info, train=is_train, text_emb=text_embedding)

    def evaluate(self, batch, n_samples, text_embedding=None, noisy_data=None, sample=True):
        pose, tp, mask = self.process_data(batch)
        side_info = self.get_side_info(tp, mask)
        samples = self.impute(pose, mask, side_info, n_samples, text_emb=text_embedding,
                              noisy_data=noisy_data, sample=sample)
        return samples, pose, (1 - mask), tp

    def get_distribution(self, text_emb, batch, t, noisy_data=None):
        """計算並回傳 Normal 分佈 (用於 GRPO single-step)。

        Args:
            noisy_data: 若提供則直接使用，否則自動從 t 生成雜訊。
        """
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        # shape: (B, K, L) — K=joints*3, L=frames
        B = observed_data.shape[0]
        frames = observed_data.shape[2]   # L
        joints = observed_data.shape[1] // 3  # K/3

        t = t.to(self.alpha_torch.device)

        if noisy_data is None:
            current_alpha = self.alpha_torch[t].to(observed_data.device)
            noise = torch.randn_like(observed_data)
            noisy_data = current_alpha.sqrt() * observed_data + (1.0 - current_alpha).sqrt() * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, gt_mask)
        pred = self.diffmodel(total_input, side_info, t, text_emb=text_emb)  # (B, K, L)

        # reshape to (B, frames, joints, 3) for GRPO reward computation
        pred = pred.permute(0, 2, 1)              # (B, L, K)
        pred = pred.reshape(B, frames, joints, 3) # (B, frames, joints, 3)

        mean = pred
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        std = torch.ones_like(mean) * self.sampling_std
        return torch.distributions.Normal(mean, std)

    def get_n_log_prob(self, dist, samples):
        """計算 G 個 samples 在 dist 下的 log-prob，用於 GRPO policy loss。

        Args:
            dist   : Normal(mean (B, frames, joints, 3), std)
            samples: (B, G, frames, joints, 3)

        Returns:
            (B, G) 平均 log prob
        """
        mean = dist.loc.unsqueeze(1)   # (B, 1, frames, joints, 3)
        std  = dist.scale.unsqueeze(1)
        dist_exp = torch.distributions.Normal(mean, std)
        log_prob = dist_exp.log_prob(samples.detach())  # (B, G, frames, joints, 3)
        num_elements = samples.shape[-1] * samples.shape[-2] * samples.shape[-3]
        return log_prob.sum(dim=(-3, -2, -1)) / num_elements  # (B, G)

    def sample_n(self, text_emb, batch, t, G):
        """Single-step: 生成 G 個 samples 及對應 distribution。

        Returns:
            samples: (B, G, frames, joints, 3)
            dist   : Normal
        """
        dist = self.get_distribution(text_emb, batch, t)
        samples = dist.rsample((G,)).permute(1, 0, 2, 3, 4)  # (B, G, frames, joints, 3)
        return samples, dist

    # ====================== 核心模組 ==========================

    # ---- R3 (Strategy C) helpers ----

    def _iter_moe_blocks(self):
        """Yield (key, moe_block) for every FairscaleMoEBlock(_RL) in diffmodel.

        Key format: ``"time_R{r}_L{i}"`` / ``"feature_R{r}_L{i}"``.
        """
        for r, res_block in enumerate(self.diffmodel.residual_layers):
            for prefix, encoder in [("time", getattr(res_block, "time_layer", None)),
                                    ("feature", getattr(res_block, "feature_layer", None))]:
                if encoder is None or not hasattr(encoder, "layers"):
                    continue
                for i, enc_layer in enumerate(encoder.layers):
                    moe_block = getattr(enc_layer, "moe_layer", None)
                    if moe_block is not None:
                        yield f"{prefix}_R{r}_L{i}", moe_block

    def _collect_moe_routing(self):
        """Read ``_last_selected_experts`` from every MoE block (after a forward pass).

        Returns:
            dict[str, Tensor]: key → (N, 2) int64, **detached & on CPU** to avoid
            keeping stale CUDA graph references.
        """
        routing = {}
        for key, block in self._iter_moe_blocks():
            sel = getattr(block, "_last_selected_experts", None)
            if sel is not None:
                routing[key] = sel.detach()  # stay on same device for injection
        return routing

    def _inject_moe_routing(self, routing):
        """Set ``_pending_routing`` on every FairscaleMoEBlock_RL for the next forward.

        Args:
            routing: dict returned by ``_collect_moe_routing()``.
        """
        for key, block in self._iter_moe_blocks():
            if key in routing and hasattr(block, "set_pending_routing"):
                block.set_pending_routing(routing[key])

    def _calc_loss_r3(self, x0, cond_mask, side, text_emb=None):
        """Strategy C — Detached Single‑Step R3 loss.

        Two forward passes on the **same** (x_t, t):
          Pass 1 (no_grad): run diffmodel to record routing decisions.
          Pass 2 (with grad): replay routing via FairscaleMoEBlock_RL masked softmax.

        Compute overhead ≈ +1× forward (no backward on Pass 1).
        """
        B, K, L = x0.shape
        t = torch.randint(0, self.num_steps, (B,), device=x0.device)
        cur_alpha = self.alpha_torch.to(x0.device)[t]

        noise = torch.randn_like(x0)
        x_t = cur_alpha.sqrt() * x0 + (1 - cur_alpha).sqrt() * noise
        inp = self.set_input_to_diffmodel(x_t, x0, cond_mask)

        # --- Pass 1: collect I_infer (detached) ---
        with torch.no_grad():
            self.diffmodel(inp, side, t, text_emb=text_emb)
        routing = self._collect_moe_routing()

        # --- Pass 2: R3 replay (with grad) ---
        self._inject_moe_routing(routing)
        pred_eps = self.diffmodel(inp, side, t, text_emb=text_emb)

        # --- Loss ---
        reconstruction_loss = ((noise - pred_eps) * (1 - cond_mask)).pow(2).sum()
        denom = (1 - cond_mask).sum().clamp(min=1)
        reconstruction_loss = reconstruction_loss / denom
        total_loss = reconstruction_loss

        # MoE load balancing loss
        if self.balancing_loss:
            if hasattr(self.diffmodel, "get_load_balancing_loss"):
                load_balancing_loss, _total_fi = self.diffmodel.get_load_balancing_loss()
                total_loss += self.balance_loss_weight * load_balancing_loss

        self.time_step += 1
        return total_loss

    def get_side_info(self, tp, cond_mask):
        B, K, L = cond_mask.shape
        t_emb = self._time_emb(tp, self.emb_time_dim).unsqueeze(2).expand(-1, -1, K, -1)
        f_emb = self.embed_layer(torch.arange(self.target_dim, device=cond_mask.device)).unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side = torch.cat([t_emb, f_emb], dim=-1).permute(0, 3, 2, 1)  # (B, C, K, L)
        if not self.is_unconditional:
            side = torch.cat([side, cond_mask.unsqueeze(1)], dim=1)
        return side

    def set_input_to_diffmodel(self, x_t, x0, cond_mask):
        if self.is_unconditional:
            return x_t.unsqueeze(1)
        obs = (cond_mask * x0).unsqueeze(1)
        no_tg = ((1 - cond_mask) * x_t).unsqueeze(1)
        return torch.cat([obs, no_tg], dim=1)

    def _calc_loss(self, x0, cond_mask, side, train=True, text_emb=None, set_t=-1):
        B, K, L = x0.shape
        t = torch.randint(0, self.num_steps, (B,), device=x0.device) if train else torch.full((B,), set_t, dtype=torch.long, device=x0.device)
        cur_alpha = self.alpha_torch.to(x0.device)[t]

        noise = torch.randn_like(x0)
        x_t = cur_alpha.sqrt() * x0 + (1 - cur_alpha).sqrt() * noise

        inp = self.set_input_to_diffmodel(x_t, x0, cond_mask)
        pred_eps = self.diffmodel(inp, side, t, text_emb=text_emb)
        # exit(0)

        # 主要的重建損失
        reconstruction_loss = ((noise - pred_eps) * (1 - cond_mask)).pow(2).sum()
        denom = (1 - cond_mask).sum().clamp(min=1)
        reconstruction_loss = reconstruction_loss / denom
        total_loss = reconstruction_loss
        # MoE load balancing loss (如果存在)
        if self.balancing_loss:
            if hasattr(self.diffmodel, 'get_load_balancing_loss'):
                load_balancing_loss, _total_fi = self.diffmodel.get_load_balancing_loss()
                total_loss += self.balance_loss_weight * load_balancing_loss
                
        self.time_step += 1
        return total_loss

    def impute(self, x0, cond_mask, side, n, text_emb=None, noisy_data=None, sample=True):
        """Full denoising chain.

        Args:
            noisy_data: (B, K, L) or (B, n, K, L) — 若給定則以此為初始雜訊。
            sample    : 是否在每步加入隨機雜訊（True = 標準 DDPM）。
        """
        B, K, L = x0.shape
        device = x0.device
        alpha_hat = torch.as_tensor(self.alpha_hat, device=device, dtype=x0.dtype)
        alpha = torch.as_tensor(self.alpha, device=device, dtype=x0.dtype)
        beta = torch.as_tensor(self.beta, device=device, dtype=x0.dtype)

        if noisy_data is not None and noisy_data.ndim == 4 and noisy_data.shape[1] == n:
            x_t = noisy_data.to(device=device, dtype=x0.dtype).clone()
        elif noisy_data is not None and noisy_data.ndim == 3:
            x_t = noisy_data.to(device=device, dtype=x0.dtype).unsqueeze(1).expand(-1, n, -1, -1).clone()
        else:
            x_t = torch.randn(B, n, K, L, device=device, dtype=x0.dtype)

        side_rep = side.repeat_interleave(n, dim=0)
        cond_rep = cond_mask.repeat_interleave(n, dim=0)
        obs_rep = x0.repeat_interleave(n, dim=0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(n, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(n, dim=0) if text_emb is not None else None

        for t in reversed(range(self.num_steps)):
            x_t_flat = x_t.reshape(B * n, K, L)
            inp = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * n,), t, dtype=torch.long, device=device)
            eps = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)

            coeff1 = 1 / alpha_hat[t].sqrt()
            coeff2 = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
            
            if sample:
                # DDPM
                mean = coeff1 * (x_t_flat - coeff2 * eps)
                if t > 0:
                    sigma = ((1 - alpha[t - 1]) / (1 - alpha[t]) * beta[t]).sqrt()
                    
                    z = torch.randn_like(x_t_flat)
                    x_t_flat = mean + sigma * z
                else:
                    x_t_flat = mean
            else:
                # DDIM (deterministic)
                alpha_bar_t = alpha[t]
                alpha_bar_prev = alpha[t - 1] if t > 0 else torch.tensor(1.0, device=device)
                
                # Predict x0 from x_t and eps
                pred_x0 = (x_t_flat - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
                
                # Direction pointing to x_t
                dir_xt = (1 - alpha_bar_prev).sqrt() * eps
                
                x_t_flat = alpha_bar_prev.sqrt() * pred_x0 + dir_xt

            x_t = x_t_flat.reshape(B, n, K, L)

        return x_t * (1 - cond_mask.unsqueeze(1)) + x0.unsqueeze(1) * cond_mask.unsqueeze(1)

    # ====================== GRPO / Trajectory ==========================

    def sample_trajectory(self, text_emb, batch, G, return_step_log_probs=False, active_timesteps=None, diffmodel_hook=None):
        """全去噪鏈取樣，回傳 G 條軌跡供 GRPO 使用。

        當 ``use_r3=True`` 時，額外收集每步的 MoE routing 決策，
        供 ``backprop_trajectory_loss`` 在 Pass 2 做 R3 replay。

        Returns:
            final_samples  : (B, G, K, L)
            total_log_probs: (B, G), future-only normalized trajectory log-prob
            step_log_probs : optional (num_steps-1, B, G), returned when
                             return_step_log_probs=True
            all_latents    : list[Tensor (B, G, K, L)], length = num_steps+1
                             [x_T, x_{T-1}, ..., x_0]
            all_routing    : list[dict] | None
                             若 ``use_r3``，長度 = num_steps，迭代順序同 loop。
                             index 0 = t=T-1，最後一筆 = deterministic t=0；
                             log-prob / backprop 只使用 t=T-1...1。
                             每個 dict: key → (N, 2) int64 detached tensor。
                             若非 R3 模式，回傳 ``None``。
        """
        if active_timesteps is not None:
            active_timesteps = set(active_timesteps)
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape
        device = observed_data.device

        noise_scale = max(float(getattr(self, "sampling_std", 1.0)), 1e-6)
        x_t = (torch.randn(B, 1, K, L, device=device) * noise_scale).expand(-1, G, -1, -1).clone()
        all_latents = [x_t]
        total_log_probs = torch.zeros(B, G, device=device)
        step_log_probs = []
        all_routing = [] if self.use_r3 else None

        alpha_hat = torch.tensor(self.alpha_hat, device=device).float()
        alpha = torch.tensor(self.alpha, device=device).float()

        side_rep  = side_info.repeat_interleave(G, dim=0)
        cond_rep  = gt_mask.repeat_interleave(G, dim=0)
        obs_rep   = observed_data.repeat_interleave(G, dim=0)
        future_mask = (1 - gt_mask).repeat_interleave(G, dim=0)
        future_denom = future_mask.sum(dim=(1, 2)).clamp(min=1.0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0) if text_emb is not None else None

        for t in reversed(range(self.num_steps)):
            x_t_flat = x_t.reshape(B * G, K, L)
            inp = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t, dtype=torch.long, device=device)

            if diffmodel_hook is not None:
                eps = diffmodel_hook(self.diffmodel, t, inp, side_rep, t_vec, text_rep)
            else:
                eps  = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)

            # R3: 收集本步 routing（推理 Pass 1）
            if self.use_r3:
                all_routing.append(self._collect_moe_routing())

            c1   = 1 / alpha_hat[t].sqrt()
            c2   = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
            mean = c1 * (x_t_flat - c2 * eps)

            if t > 0:
                # Check if this timestep should be tracked for gradients/log-probs
                is_active = (t in active_timesteps) if active_timesteps is not None else True
                
                if is_active:
                    sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t]) * noise_scale
                    z = torch.randn_like(mean)
                    x_prev = mean + sigma * z
                    step_lp_elem = -0.5 * z ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
                    step_lp = (step_lp_elem * future_mask).sum(dim=(1, 2)) / future_denom
                    step_lp = step_lp.reshape(B, G)
                    total_log_probs += step_lp
                    step_log_probs.append(step_lp)
                else:
                    # Outside the window: Shared noise across G variants (Shared Noise DDPM)
                    shared_noise_scale = float(getattr(self, "shared_noise_scale", 1.0))
                    sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t]) * shared_noise_scale
                    z_shared = torch.randn(B, 1, K, L, device=device).expand(-1, G, -1, -1).reshape(B * G, K, L)
                    z = z_shared
                    x_prev = mean + sigma * z
                    
                x_t = x_prev.reshape(B, G, K, L)
            else:
                x_t = mean.reshape(B, G, K, L)

            all_latents.append(x_t)
            # 關鍵：截斷梯度，避免深度為 T 的計算圖
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
        """Return per-denoising-step log-probs for a sampled trajectory.

        The deterministic final transition ``t=0`` is excluded. Each step is
        averaged over future/missing elements only, so the scale is independent
        of the observed prefix, joint count, and sequence length.
        """
        if active_timesteps is not None:
            active_timesteps = set(active_timesteps)
        observed_data, observed_tp, gt_mask = self.process_data(batch)
        side_info = self.get_side_info(observed_tp, gt_mask)
        B, K, L = observed_data.shape
        G = all_latents[0].shape[1]
        device = observed_data.device

        alpha_hat = torch.tensor(self.alpha_hat, device=device).float()
        alpha     = torch.tensor(self.alpha,     device=device).float()

        side_rep = side_info.repeat_interleave(G, dim=0)
        cond_rep = gt_mask.repeat_interleave(G, dim=0)
        obs_rep  = observed_data.repeat_interleave(G, dim=0)
        future_mask = (1 - gt_mask).repeat_interleave(G, dim=0)
        future_denom = future_mask.sum(dim=(1, 2)).clamp(min=1.0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0) for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0) if text_emb is not None else None

        noise_scale = max(float(getattr(self, "sampling_std", 1.0)), 1e-6)
        step_lps = []
        for t in reversed(range(1, self.num_steps)):
            is_active = (t in active_timesteps) if active_timesteps is not None else True
            if not is_active:
                continue

            idx_curr = self.num_steps - (t + 1)
            idx_next = self.num_steps - t
            x_t    = all_latents[idx_curr]
            x_prev = all_latents[idx_next]

            x_t_flat = x_t.reshape(B * G, K, L).detach()
            inp  = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t, dtype=torch.long, device=device)

            routing_idx = self.num_steps - 1 - t
            if all_routing is not None and routing_idx < len(all_routing):
                self._inject_moe_routing(all_routing[routing_idx])

            eps  = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)
            c1   = 1 / alpha_hat[t].sqrt()
            c2   = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
            mean = c1 * (x_t_flat - c2 * eps)

            sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t]) * noise_scale
            x_prev_flat = x_prev.reshape(B * G, K, L).detach()
            mse  = (x_prev_flat - mean) ** 2
            lp_elem = -0.5 * mse / sigma ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
            lp = (lp_elem * future_mask).sum(dim=(1, 2)) / future_denom
            step_lps.append(lp.reshape(B, G))

        if not step_lps:
            return torch.empty(0, B, G, device=device)
        return torch.stack(step_lps, dim=0)

    def get_trajectory_log_prob(self, all_latents, text_emb, batch):
        """計算給定軌跡在本模型（作為 reference）下的 log-prob。

        Args:
            all_latents: [x_T, ..., x_0], len = num_steps+1, each (B, G, K, L)

        Returns:
            (B, G) total log prob
        """
        step_lps = self.get_trajectory_step_log_probs(all_latents, text_emb, batch)
        if step_lps.numel() == 0:
            B = all_latents[0].shape[0]
            G = all_latents[0].shape[1]
            return torch.zeros(B, G, device=all_latents[0].device)
        return step_lps.mean(dim=0)

    def backprop_trajectory_loss(self, all_latents, text_emb, batch, advantages,
                                  old_step_log_probs=None,
                                  ref_step_log_probs=None,
                                  epsilon=0.2,
                                  kl_coef=0.0,
                                  max_kl_penalty=None,
                                  all_routing=None,
                                  active_timesteps=None,
                                  loss_scale=1.0,
                                  backward_fn=None,
                                  ablation_cfg=None,
                                  pl_module=None):
        """Memory-efficient clipped GRPO/PPO backprop.
        【記憶體拯救黑科技：逐步釋放梯度的工程實作】
        正常來說，如果要反向傳播 50 步去噪過程，神經網路 (U-Net) 需要把 50 步的所有 Forward 
        Activation (特徵圖) 全部存在顯存 (VRAM) 裡，這絕對會 OOM。
        這裡我們用的「工程技巧`」是：
        1. 在 forward (採樣) 時，使用 `x_t.detach()` 把每一步的計算圖切斷，只存下各步的數值。
        2. 在這裡 backward 時，我們用 for 迴圈一次只重算一個 $t$ 步 of U-Net。
        3. 算完這一步的 loss，**立刻呼叫 `.backward()`**。PyTorch 會把梯度累積在 `.grad` 中。
        4. 然後 `del loss_step, ...` 清除這一步的計算圖與記憶體。
        因為 Loss 總和的梯度 = 梯度的總和，這樣做我們只需消耗「1 次」U-Net forward 的顯存，
        卻能完美算出 50 步的聯合梯度！
        """
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

        alpha_hat = torch.tensor(self.alpha_hat, device=device).float()
        alpha     = torch.tensor(self.alpha,     device=device).float()

        # detach() 是關鍵！確保條件輸入不會不小心拉起不需要的計算圖
        side_rep = side_info.repeat_interleave(G, dim=0).detach()
        cond_rep = gt_mask.repeat_interleave(G, dim=0).detach()
        obs_rep  = observed_data.repeat_interleave(G, dim=0).detach()
        future_mask = (1 - gt_mask).repeat_interleave(G, dim=0).detach()
        future_denom = future_mask.sum(dim=(1, 2)).clamp(min=1.0)
        if isinstance(text_emb, tuple):
            text_rep = tuple(u.repeat_interleave(G, dim=0).detach() for u in text_emb)
        else:
            text_rep = text_emb.repeat_interleave(G, dim=0).detach() if text_emb is not None else None

        adv = advantages.detach()
        if ablation_cfg is not None:
            topk_cfg = ablation_cfg.get("topk_gating", {})
            if topk_cfg.get("enabled", False):
                ratio = topk_cfg.get("ratio", 0.5)
                k = max(1, int(G * ratio))
                topk_vals, _ = torch.topk(adv, k, dim=1)
                min_topk = topk_vals[:, -1:]
                gate_mask = (adv >= min_topk).float()
                adv = adv * gate_mask

        noise_scale = max(float(getattr(self, "sampling_std", 1.0)), 1e-6)
        active_steps = len(active_timesteps) if active_timesteps is not None else (self.num_steps - 1)

        active_step_idx = 0
        # 由 $t=50$ 倒推回 $t=1$，逐步計算並馬上反向傳播 (Backprop)
        for t in reversed(range(1, self.num_steps)):
            is_active = (t in active_timesteps) if active_timesteps is not None else True
            if not is_active:
                continue

            idx_curr = self.num_steps - (t + 1)
            idx_next = self.num_steps - t
            x_t    = all_latents[idx_curr]
            x_prev = all_latents[idx_next]

            # 把採樣時存下來的 $x_t$ 拿出來，務必 detach() 確保沒有以前的計算圖
            x_t_flat = x_t.reshape(B * G, K, L).detach()
            inp  = self.set_input_to_diffmodel(x_t_flat, obs_rep, cond_rep)
            t_vec = torch.full((B * G,), t, dtype=torch.long, device=device)

            routing_idx = self.num_steps - 1 - t
            if all_routing is not None and routing_idx < len(all_routing):
                self._inject_moe_routing(all_routing[routing_idx])

            # 重新執行一次當前 step 的 U-Net 推論！(只有這裡會消耗 VRAM 並建立局部的計算圖)
            eps  = self.diffmodel(inp, side_rep, t_vec, text_emb=text_rep)
            
            # 利用網路預測的 eps，算出 $x_{t-1}$ 的高斯分佈平均值 (mean)
            c1   = 1 / alpha_hat[t].sqrt()
            c2   = (1 - alpha_hat[t]) / (1 - alpha[t]).sqrt()
            mean = c1 * (x_t_flat - c2 * eps)

            sigma = math.sqrt((1 - alpha[t - 1]) / (1 - alpha[t]) * self.beta[t]) * noise_scale
            x_prev_flat = x_prev.reshape(B * G, K, L).detach()
            
            # --- 以下是計算 Log-Probability ---
            # mse 代表神經網路預測的 mean，和「當初採樣時實際加上雜訊走的那條路 (x_prev)」的差距
            mse  = (x_prev_flat - mean) ** 2
            
            # 這是標準常態分佈 Log-Prob 的數學公式：-0.5 * (x - mu)^2 / sigma^2 - log(sigma) ...
            lp_elem = -0.5 * mse / sigma ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
            
            # 只針對我們要生成的目標部分 (future_mask) 算機率，除以數量做正規化
            lp = (lp_elem * future_mask).sum(dim=(1, 2)) / future_denom
            new_lp = lp.reshape(B, G)

            # --- 計算 PPO / GRPO Surrogate Loss ---
            adv_t = adv[active_step_idx] if adv.dim() == 3 else adv
            if ablation_cfg is not None:
                decay_cfg = ablation_cfg.get("step_decay_advantage", {})
                if decay_cfg.get("enabled", False):
                    power = decay_cfg.get("power", 1.0)
                    decay_weight = (t / self.num_steps) ** power
                    adv_t = adv_t * decay_weight

            if old_step_log_probs is None:
                # 單純的 Policy Gradient: Loss = -LogProb * Advantage
                loss_step = -(new_lp * adv_t).mean()
            else:
                # PPO 裁切機制 (Clipped Surrogate Objective)
                old_lp = old_step_log_probs[active_step_idx].to(device).detach()
                ratio = torch.exp(torch.clamp(new_lp - old_lp, min=-20.0, max=20.0))
                
                # 限制更新幅度在 [1-epsilon, 1+epsilon] 之間，防止一次更新把模型搞崩
                clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
                
                # 取較小的那個作為 surrogate loss
                surrogate = torch.minimum(ratio * adv_t, clipped_ratio * adv_t)
                loss_step = -surrogate.mean()

            # --- 加入 KL 散度懲罰 (KL Penalty) ---
            if ref_step_log_probs is not None and kl_coef > 0:
                ref_lp = ref_step_log_probs[active_step_idx].to(device).detach()
                log_ref_new = torch.clamp(ref_lp - new_lp, min=-20.0, max=20.0)
                approx_kl = torch.exp(log_ref_new) - log_ref_new - 1.0
                kl_loss = kl_coef * approx_kl
                if max_kl_penalty is not None:
                    kl_loss = torch.clamp(kl_loss, max=max_kl_penalty)
                loss_step = loss_step + kl_loss.mean()

            # 平均攤提這 active_steps 步的 loss
            loss_step = (loss_step / active_steps) * loss_scale
            
            # Check if this is the last active step in the loop
            is_last_step = (active_step_idx == active_steps - 1)
            
            # ⭐️ 【核心技巧】：在迴圈內直接 Backward！
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

            # ⭐️ 【釋放記憶體】：把所有跟計算圖有關的暫存變數通通砍掉！
            del loss_step, lp, new_lp, eps, mean, inp
            active_step_idx += 1

    # ====================== 其他輔助 ==========================

    def process_data(self, batch):
        device = self.current_device
        pose = batch["pose"].to(device).float().permute(0, 2, 1)
        #print("pose shape:", pose.shape)
        #exit(0)
        tp = batch["timepoints"].to(device).float()
        mask = batch["mask"].to(device).float().permute(0, 2, 1)
        return pose, tp, mask

    def _time_emb(self, pos, d_model):
        pe = torch.zeros_like(pos).unsqueeze(-1).repeat(1, 1, d_model)
        div = torch.exp(torch.arange(0, d_model, 2, device=pos.device) * (-math.log(10000.0) / d_model))
        pe[..., 0::2] = torch.sin(pos.unsqueeze(-1) * div)
        pe[..., 1::2] = torch.cos(pos.unsqueeze(-1) * div)
        return pe

    def _make_beta_schedule(self, cfg_diff):
        schedule = cfg_diff["schedule"]
        if schedule == "quad":
            return np.linspace(cfg_diff["beta_start"] ** 0.5, cfg_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif schedule == "linear":
            return np.linspace(cfg_diff["beta_start"], cfg_diff["beta_end"], self.num_steps)
        elif schedule == "cosine":
            return self._betas_for_alpha_bar(
                self.num_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def _betas_for_alpha_bar(self, T, alpha_bar, max_beta=0.5):
        return np.array([min(1 - alpha_bar((i + 1) / T) / alpha_bar(i / T), max_beta) for i in range(T)])
