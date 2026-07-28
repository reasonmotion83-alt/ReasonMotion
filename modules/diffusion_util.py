import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.moe_transformer import MoETransformerEncoderLayer

def get_torch_trans(heads=8, layers=1, channels=64):
    #print("base torch transformer")
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_torch_trans_moe(heads=8, layers=1, channels=64, num_experts=4, top_k=2, moe_type: str = "qwen"):
    """Create a TransformerEncoder with MoE FFN.

    moe_type:
        - "qwen": 使用原本自訂的 Qwen-style MoE (舊版行為)
        - "tutel": 使用 Tutel 的 moe_layer 實作
    """
    print(f"MoE torch transformer (type={moe_type})")
    encoder_layer = MoETransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
        num_experts=num_experts,
        top_k=top_k,
        moe_type=moe_type,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        if diffusion_step.is_floating_point():
            device = self.embedding.device
            diffusion_step = diffusion_step.to(device)
            max_step = float(self.embedding.shape[0] - 1)
            diffusion_step = torch.clamp(diffusion_step, 0.0, max_step)
            
            dim = self.embedding.shape[1] // 2
            frequencies = 10.0 ** (torch.arange(dim, device=device, dtype=diffusion_step.dtype) / (dim - 1) * 4.0).unsqueeze(0)
            steps = diffusion_step.unsqueeze(1)
            table = steps * frequencies
            x = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        else:
            diffusion_step = diffusion_step.to(self.embedding.device).long()
            diffusion_step = torch.clamp(diffusion_step, 0, self.embedding.shape[0] - 1)
            x = self.embedding[diffusion_step]
            
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2, transformer_fn=None, text_mode="token"):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    textemb=config["textemb"],
                    text_mode=text_mode,
                    moe=config.get("moe", False),
                    temporal_decay=config.get("temporal_decay", False),
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, text_emb=None):
        # 1, 2, 72, 70
        # Channel 0 → 有觀測資料（GT）
		# Channel 1 → 要預測區域（含噪）

        B, inputdim, K, L = x.shape
        # print("x shape:", x.shape)
        # print("x : ", x[:,0,:, 30:])  # 移到 CPU 且轉 numpy
        
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, text_emb=text_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        return x.reshape(B, K, L)

    def get_load_balancing_loss(self):
        """
        收集所有 MoE 層的 load balancing loss
        """
        total_loss = 0.0
        count = 0
        total_fi = None  # 延遲初始化，自動匹配 num_experts 與 device
        for layer in self.residual_layers:
            if hasattr(layer, 'time_layer') and hasattr(layer.time_layer, 'layers'):
                for transformer_layer in layer.time_layer.layers:
                    if hasattr(transformer_layer, 'get_load_balancing_loss'):
                        loss, fi = transformer_layer.get_load_balancing_loss()
                        if total_fi is None:
                            total_fi = torch.zeros_like(fi)
                        total_loss += loss
                        total_fi += fi
                        count += 1
            
            if hasattr(layer, 'feature_layer') and hasattr(layer.feature_layer, 'layers'):
                for transformer_layer in layer.feature_layer.layers:
                    if hasattr(transformer_layer, 'get_load_balancing_loss'):
                        loss, fi = transformer_layer.get_load_balancing_loss()
                        if total_fi is None:
                            total_fi = torch.zeros_like(fi)
                        total_loss += loss
                        total_fi += fi
                        count += 1

        if count == 0:
            # 沒有 MoE 層，回傳零損失
            return torch.tensor(0.0), torch.zeros(1)

        return total_loss / count, total_fi / count

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, textemb=384, text_mode="cross_attn", moe=False, temporal_decay=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.text_mode = text_mode
        if text_mode == "cross_attn" or text_mode == "decoupled_cross_attn":
            self.text_proj = nn.Linear(textemb, channels)
            self.cross_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=nheads, batch_first=True)
        elif text_mode == "adaln":
            self.adaln_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(textemb, 2 * channels)
            )
            self.norm = nn.LayerNorm(channels)

        # moe 參數向後相容：
        #   - False / 0  : 完全不用 MoE（舊行為）
        #   - True / 1   : 使用舊版 Qwen-style MoE
        #   - "qwen"     : 明確指定 Qwen-style MoE
        #   - "tutel"    : 使用 Tutel 實作的 MoE
        #   - "fairscale": 使用 Fairscale 的 GShard Top-2 MoE
        moe_type = "qwen"
        use_moe = False
        if isinstance(moe, bool):
            use_moe = moe
        elif isinstance(moe, str):
            moe_type = moe
            use_moe = True
        else:
            # 其他數值型別，非 0 視為開啟 MoE
            try:
                use_moe = bool(moe)
            except Exception:
                use_moe = False

        if not use_moe:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans_moe(
                heads=nheads,
                layers=1,
                channels=channels,
                num_experts=16,
                top_k=2,
                moe_type=moe_type,
            )
            self.feature_layer = get_torch_trans_moe(
                heads=nheads,
                layers=1,
                channels=channels,
                num_experts=16,
                top_k=2,
                moe_type=moe_type,
            )
        # causal mask for Transformer (L, L)
        # 70 is a sufficiently large sequence length upper bound
      
        self.time_mask = torch.triu(torch.ones(200, 200) * float('-inf'), diagonal=1)

        # --- Temporal Decay Mask (GAtFuN-inspired logistic decay) ---
        # 透過 additive mask log(σ(β - α|i-j|)) = -softplus(α|i-j| - β)
        # 讓 temporal attention 偏好相近幀，α 控制衰減陡度，β 控制偏移
        self.temporal_decay = temporal_decay
        if temporal_decay:
            self.decay_alpha = nn.Parameter(torch.tensor(1.0))
            self.decay_beta = nn.Parameter(torch.tensor(5.0))

    def forward_time(self, y, base_shape, text_emb=None):
        B, C, K, L = base_shape

        if L == 1:
            return y
        
        # (B, C, K, L) -> (B*K, L, C)
        y = y.reshape(B, C, K, L).permute(0, 2, 3, 1).reshape(B * K, L, C)
        
        # --- Decoupled Cross Attention ---
        if self.text_mode == "decoupled_cross_attn" and text_emb is not None:
            tok_emb, tok_mask = text_emb
            mem = self.text_proj(tok_emb)  # (B, num_tokens, C)
            # 展開以對應 B*K
            mem = mem.unsqueeze(1).expand(B, K, -1, C).reshape(B * K, -1, C)
            mask_exp = tok_mask.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            
            attn_out, _ = self.cross_attn(y, mem, mem, key_padding_mask=~mask_exp.bool())
            y = y + attn_out
        # ---------------------------------

        if self.temporal_decay:
            indices = torch.arange(L, device=y.device, dtype=y.dtype)
            dist_matrix = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()  # (L, L)
            decay_mask = -F.softplus(self.decay_alpha * dist_matrix - self.decay_beta)
            y = self.time_layer(y, mask=decay_mask)
        else:
            y = self.time_layer(y)

        # (B*K, L, C) -> (B, C, K*L)
        return y.reshape(B, K, L, C).permute(0, 3, 1, 2).reshape(B, C, K * L)

    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y

        # (B, C, K, L) -> (B*L, K, C)
        y = y.reshape(B, C, K, L).permute(0, 3, 2, 1).reshape(B * L, K, C)
        y = self.feature_layer(y)
        # (B*L, K, C) -> (B, C, K*L)
        y = y.reshape(B, L, K, C).permute(0, 3, 2, 1).reshape(B, C, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, text_emb=None):
        B, C, K, L = x.shape
        base_shape = x.shape

        y = x.reshape(B, C, K * L)

        # --- Adaptive LayerNorm ---
        if self.text_mode == "adaln" and text_emb is not None:
            tok_emb, tok_mask = text_emb
            # Pool token embeddings (mean over unmasked tokens)
            mask_float = tok_mask.float().unsqueeze(-1)
            pooled_text = (tok_emb * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-5)
            # Get shift and scale
            shift, scale = self.adaln_proj(pooled_text).chunk(2, dim=1) # (B, C)
            
            # Apply AdaLN
            y_norm = self.norm(y.permute(0, 2, 1)).permute(0, 2, 1) # (B, C, K*L)
            y = y_norm * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
            # Add diffusion embedding AFTER AdaLN so its mean shift is not erased
            y = y + self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        else:
            # Add diffusion embedding FIRST for other modes
            y = y + self.diffusion_projection(diffusion_emb).unsqueeze(-1)
            
            # --- Original Cross Attention ---
            if self.text_mode == "cross_attn" and text_emb is not None:
                tok_emb, tok_mask = text_emb
                mem = self.text_proj(tok_emb)
                q = y.permute(0, 2, 1)
                attn_out, _ = self.cross_attn(q, mem, mem, key_padding_mask=~tok_mask.bool())
                y = q + attn_out
                y = y.permute(0, 2, 1)

        y = self.forward_time(y, base_shape, text_emb)
        y = self.forward_feature(y, base_shape)

        y = self.mid_projection(y)
        cond_info = cond_info.reshape(B, cond_info.shape[1], K * L)
        y = y + self.cond_projection(cond_info)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual.reshape(base_shape)) / math.sqrt(2.0), skip.reshape(base_shape)