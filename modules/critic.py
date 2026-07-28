import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionCriticNetwork(nn.Module):
    """
    Spatial-Temporal Value Network V_phi(x_t, t, c) for DDPO on 3D Motion Diffusion.
    Estimates scalar expected trajectory return from intermediate noisy latents x_t.
    """
    def __init__(self, target_dim: int = 72, timeemb_dim: int = 128, textemb_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.target_dim = target_dim
        self.timeemb_dim = timeemb_dim
        
        # 1D Spatial-Temporal Convolution over motion joints & sequence length
        self.motion_proj = nn.Sequential(
            nn.Conv1d(target_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1) # Pool temporal dimension -> (N, hidden_dim, 1)
        )
        
        # Timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(timeemb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Text Prompt MLP
        self.text_mlp = nn.Sequential(
            nn.Linear(textemb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _timestep_embedding(self, timesteps, dim):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=timesteps.device)
        args = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x_t, t, text_emb=None):
        """
        Args:
            x_t: Noisy motion latents (N, K, L)
            t: Timesteps tensor (N,)
            text_emb: Text embedding tuple (tok_emb, tok_mask) or Tensor (N, textemb_dim)
        Returns:
            Scalar estimated values (N, 1)
        """
        N, K, L = x_t.shape
        
        # Motion feature
        h_motion = self.motion_proj(x_t).squeeze(-1) # (N, hidden_dim)
        
        # Time feature
        t_emb = self._timestep_embedding(t, self.timeemb_dim)
        h_time = self.time_mlp(t_emb) # (N, hidden_dim)
        
        # Text feature
        if isinstance(text_emb, tuple):
            tok_emb, tok_mask = text_emb
            mask_expanded = tok_mask.unsqueeze(-1).float()
            pooled_text = (tok_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)
            h_text = self.text_mlp(pooled_text)
        elif text_emb is not None and torch.is_tensor(text_emb):
            if text_emb.dim() == 3:
                pooled_text = text_emb.mean(dim=1)
            else:
                pooled_text = text_emb
            h_text = self.text_mlp(pooled_text)
        else:
            h_text = 0.0

        # Fusion & Value prediction
        h_combined = h_motion + h_time + h_text
        val = self.value_head(h_combined) # (N, 1)
        return val
