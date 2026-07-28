#!/usr/bin/env python3
"""
MoE Transformer Implementation - 繼承 nn.TransformerEncoderLayer 並修改 _ff_block 為 MoE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Callable
import copy
import math

try:
    # Optional Tutel import; only required when using moe_type="tutel"
    from tutel import moe as tutel_moe
    _TUTEL_AVAILABLE = True
except ImportError:
    tutel_moe = None
    _TUTEL_AVAILABLE = False

class MoETransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    MoE Transformer Encoder Layer with specified number of experts and top-k selection.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: Union[str, Callable] = "gelu",
                 num_experts: int = 4, top_k: int = 2, moe_type: str = "qwen", **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, **kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_type = moe_type
        
        # Create MoE layer to replace standard feedforward
        if moe_type == "qwen":
            self.moe_layer = QwenStyleSparseMoEBlock(
                d_model=d_model,
                intermediate_size=dim_feedforward,
                num_experts=num_experts,
                top_k=top_k,
                norm_topk_prob=True,
                activation=activation if isinstance(activation, str) else "gelu",
                dropout=dropout
            )
        elif moe_type == "tutel":
            if not _TUTEL_AVAILABLE:
                raise ImportError(
                    "Tutel is not installed but moe_type='tutel' was requested. "
                    "Please install Tutel or use moe_type='qwen'."
                )
            self.moe_layer = TutelMoEBlock(
                d_model=d_model,
                intermediate_size=dim_feedforward,
                num_experts=num_experts,
                top_k=top_k,
                activation=activation if isinstance(activation, str) else "gelu",
                dropout=dropout
            )
        elif moe_type == "fairscale":
            self.moe_layer = FairscaleMoEBlock(
                d_model=d_model,
                intermediate_size=dim_feedforward,
                num_experts=num_experts,
                activation=activation if isinstance(activation, str) else "gelu",
                dropout=dropout
            )
        elif moe_type == "fairscale_rl":
            self.moe_layer = FairscaleMoEBlock_RL(
                d_model=d_model,
                intermediate_size=dim_feedforward,
                num_experts=num_experts,
                activation=activation if isinstance(activation, str) else "gelu",
                dropout=dropout
            )
        elif moe_type == "qwen_rl":
            self.moe_layer = QwenStyleSparseMoEBlock_RL(
                d_model=d_model,
                intermediate_size=dim_feedforward,
                num_experts=num_experts,
                top_k=top_k,
                norm_topk_prob=True,
                activation=activation if isinstance(activation, str) else "gelu",
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported MoE type: {moe_type}. Supported: 'qwen', 'qwen_rl', 'tutel', 'fairscale', 'fairscale_rl'.")

    def _ff_block(self, x: Tensor) -> Tensor:
        """
        MoE feed-forward block with top-k selection.
        """
        # print("x shape:", x.shape)
        return self.moe_layer(x)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Override to always use the slow path.

        PyTorch ≥ 2.0 ``TransformerEncoderLayer`` may use a fused C++ kernel
        (fast-path) in eval mode, which **bypasses** ``_ff_block`` entirely.
        Since our ``_ff_block`` dispatches to the MoE layer, we must ensure
        the Python path is always taken.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def get_load_balancing_loss(self) -> Tensor:
        """Get the load balancing loss from MoE layer"""
        return self.moe_layer.get_load_balancing_loss()

class QwenStyleMoEMLP(nn.Module):
    """
    Qwen-style MoE MLP expert implementation
    """
    def __init__(self, d_model: int, intermediate_size: int, activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        
        # Activation function
        if activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "relu":
            self.act_fn = F.relu
        elif activation == "silu":
            self.act_fn = F.silu
        else:
            self.act_fn = F.gelu
    
    def forward(self, x):
        """
        Forward pass through MLP expert
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class QwenStyleSparseMoEBlock(nn.Module):
    """
    Qwen-style Sparse MoE Block implementation
    """
    def __init__(self, d_model: int, intermediate_size: int = 2048, num_experts: int = 4, 
                 top_k: int = 2, norm_topk_prob: bool = True, activation: str = "gelu", 
                 dropout: float = 0.1, load_balancing_alpha: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.alpha = load_balancing_alpha
        
        # Gate network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            QwenStyleMoEMLP(d_model, intermediate_size, activation) 
            for _ in range(num_experts)
        ])
        
        # Load balancing loss
        self.load_balancing_loss = 0.0
        self.register_buffer("total_fi", torch.zeros(num_experts, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient forward pass through sparse MoE block
        """
        # print("x.shape:", hidden_states.shape)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        total_tokens = hidden_states.shape[0]
        
        # Router logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        
        # Routing weights and selected experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize top-k probabilities (Qwen3 style)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # Cast back to input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Calculate load balancing loss (pass selected_experts for top-k aware f_i)
        self.load_balancing_loss, self.total_fi = self._calculate_load_balancing_loss(router_logits, selected_experts)
        
        # Initialize final hidden states
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Memory-efficient expert processing
        # Flatten selected experts and routing weights
        flat_selected_experts = selected_experts.view(-1)  # (total_tokens * top_k,)
        flat_routing_weights = routing_weights.view(-1)    # (total_tokens * top_k,)
        
        # Create token indices for each selection
        token_indices = torch.arange(total_tokens, device=hidden_states.device)
        token_indices = token_indices.unsqueeze(1).expand(-1, self.top_k).contiguous().view(-1)
        
        # Process each expert efficiently
        for expert_idx in range(self.num_experts):
            # Find which tokens use this expert
            expert_mask = (flat_selected_experts == expert_idx)
            if not expert_mask.any():
                continue
            
            # Get tokens and weights for this expert
            expert_tokens = token_indices[expert_mask]
            expert_weights = flat_routing_weights[expert_mask]
            
            # Get inputs for this expert
            expert_inputs = hidden_states[expert_tokens]
            
            # Process through expert
            expert_outputs = self.experts[expert_idx](expert_inputs)
            
            # Apply weights
            weighted_outputs = expert_outputs * expert_weights.unsqueeze(-1)
            
            # Add to final outputs (handle potential duplicates with index_add_)
            final_hidden_states.index_add_(0, expert_tokens, weighted_outputs)
        
        # Apply dropout
        final_hidden_states = self.dropout(final_hidden_states)
        
        # Reshape back to original shape
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states
    
    def _calculate_load_balancing_loss(self, router_logits: torch.Tensor,
                                        selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Calculate load balancing loss using the Switch Transformer formula (extended to top-k).
        loss = alpha * N * sum(f_i * P_i)
        where N is the number of experts, f_i is the fraction of tokens routed to expert i
        (counting all top-k assignments), and P_i is the average router probability for expert i.

        Args:
            router_logits: (total_tokens, num_experts)
            selected_experts: (total_tokens, top_k) — the top-k expert indices per token
        """
        total_tokens = router_logits.shape[0]
        
        # P_i: average router probability for each expert
        router_probs = F.softmax(router_logits, dim=-1)
        Pi = router_probs.mean(dim=0)
        
        # f_i: fraction of token-expert assignments dispatched to expert i (top-k aware)
        flat_experts = selected_experts.view(-1)  # (total_tokens * top_k,)
        fi = torch.zeros(self.num_experts, dtype=router_probs.dtype, device=router_logits.device)
        fi.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=router_probs.dtype))
        fi = fi / (total_tokens * self.top_k)

        # Compute load balancing loss
        # The formula is N * sum(f_i * P_i), scaled by alpha
        load_loss = self.alpha * self.num_experts * torch.sum(fi * Pi)
        
        return load_loss, fi
    
    def get_load_balancing_loss(self) -> torch.Tensor:
        """Get the current load balancing loss"""
        return self.load_balancing_loss, self.total_fi


class QwenStyleSparseMoEBlock_RL(QwenStyleSparseMoEBlock):
    """R3 (Rollout Routing Replay) variant of QwenStyleSparseMoEBlock.

    與 FairscaleMoEBlock_RL 相同的 R3 機制，但基於 Qwen-style routing：
      - 推理時使用標準 top-k softmax
      - 訓練 replay 時使用 **masked softmax**：只在 I_infer 選中的
        experts 上歸一化，保留梯度流回 router

    計算流程（R3 mode）：
      1. s_train = x_train @ W_r           (router logits)
      2. I_infer = inference recorded mask  (top-k expert indices)
      3. g_replay_i = I_infer_i * exp(s_train_i)
                      / sum_j I_infer_j * exp(s_train_j)
         → masked softmax over selected experts
      4. y_replay = sum_i g_replay_i * E_i(x_train)

    使用方式同 FairscaleMoEBlock_RL（策略 C — Detached Single-Step）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_routing: Optional[torch.Tensor] = None
        # Routing replay buffers (detached, no grad)
        self._last_selected_experts: Optional[torch.Tensor] = None
        self._last_routing_weights: Optional[torch.Tensor] = None

    def set_pending_routing(self, selected_experts: torch.Tensor):
        """設定下次 forward 要 replay 的路由。

        Args:
            selected_experts: (N, top_k) int64 tensor，detached。
        """
        self._pending_routing = selected_experts

    def clear_pending_routing(self):
        """清除待用路由（安全用，forward 後也會自動清）。"""
        self._pending_routing = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """R3 forward pass for Qwen-style MoE.

        When ``_pending_routing`` is set, uses R3 masked softmax replay.
        Otherwise falls back to standard Qwen-style top-k routing.
        """
        # ---- Consume _pending_routing (Strategy C one-shot injection) ----
        external_routing = None
        if self._pending_routing is not None:
            external_routing = self._pending_routing
            self._pending_routing = None  # consumed

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)  # (N, d_model)
        N = x.shape[0]

        # Router logits (always computed — gradient source)
        router_logits = self.gate(x)  # (N, E)

        if external_routing is not None:
            # ============================================================
            # R3 Mode: Routing Replay
            # ============================================================
            selected_experts = external_routing  # (N, top_k)

            # Masked softmax: only normalize over selected experts
            infer_mask = torch.zeros(
                N, self.num_experts, device=x.device, dtype=torch.bool
            )
            infer_mask.scatter_(1, selected_experts, True)

            masked_logits = router_logits.masked_fill(~infer_mask, float("-inf"))
            gates_replay = F.softmax(masked_logits, dim=-1, dtype=torch.float32)  # (N, E)

            # Gather weights for the selected experts
            routing_weights = torch.gather(gates_replay, 1, selected_experts)  # (N, top_k)
            routing_weights = routing_weights.to(x.dtype)

            # l_aux: use full softmax for load-balancing signal
            self.load_balancing_loss, self.total_fi = self._calculate_load_balancing_loss(
                router_logits, selected_experts
            )
        else:
            # ============================================================
            # Normal / Inference Mode: standard Qwen-style top-k
            # ============================================================
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )

            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            routing_weights = routing_weights.to(x.dtype)

            self.load_balancing_loss, self.total_fi = self._calculate_load_balancing_loss(
                router_logits, selected_experts
            )

        # Cache routing for subsequent replay
        self._last_selected_experts = selected_experts.detach()
        self._last_routing_weights = routing_weights.detach()

        # ---- Sparse dispatch (same as parent) ----
        final_hidden_states = torch.zeros_like(x)
        flat_experts = selected_experts.view(-1)   # (N*top_k,)
        flat_weights = routing_weights.view(-1)    # (N*top_k,)
        token_indices = (
            torch.arange(N, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .contiguous()
            .view(-1)
        )

        for expert_idx in range(self.num_experts):
            expert_mask = (flat_experts == expert_idx)
            if not expert_mask.any():
                continue
            expert_tokens = token_indices[expert_mask]
            expert_weights = flat_weights[expert_mask]
            expert_out = self.experts[expert_idx](x[expert_tokens])
            final_hidden_states.index_add_(
                0, expert_tokens, expert_out * expert_weights.unsqueeze(-1)
            )

        final_hidden_states = self.dropout(final_hidden_states)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class TutelMoEBlock(nn.Module):
    """MoE block based on Tutel's optimized moe_layer.

    This wraps Tutel's moe_layer so that it matches the interface used by
    MoETransformerEncoderLayer and exposes a compatible get_load_balancing_loss()
    method. The old Qwen-style implementation is preserved; this is an
    additional option selected via moe_type="tutel".
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        load_balancing_alpha: float = 0.001,
    ):
        super().__init__()
        if not _TUTEL_AVAILABLE:
            raise ImportError(
                "TutelMoEBlock requires Tutel to be installed. "
                "Please install tutel and try again."
            )

        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = load_balancing_alpha

        # Activation function for Tutel's builtin FFN experts
        if activation == "gelu":
            act_fn = F.gelu
        elif activation == "relu":
            act_fn = F.relu
        elif activation == "silu":
            act_fn = F.silu
        else:
            act_fn = F.gelu

        # Tutel moe_layer: we use builtin FFN experts and top-k gate
        self.moe_layer = tutel_moe.moe_layer(
            gate_type={"type": "top", "k": top_k},
            model_dim=d_model,
            experts={
                "num_experts_per_device": num_experts,
                "type": "ffn",
                "hidden_size_per_expert": intermediate_size,
                "activation_fn": act_fn,
            },
            # Mark expert params to skip allreduce in external DDP, same as docs
            scan_expert_func=lambda name, param: setattr(param, "skip_allreduce", True),
            # Ask Tutel to return (output, aux_loss) where `output.l_aux` is aux loss
            result_func=lambda output: (output, output.l_aux),
        )

        self.dropout = nn.Dropout(dropout)
        # Load balancing loss from Tutel (aux loss). Initialized as 0.
        self.load_balancing_loss = torch.tensor(0.0)
        # Per-expert usage fraction (not directly exposed by Tutel).
        # We keep a buffer of zeros so the interface stays compatible.
        self.register_buffer("total_fi", torch.zeros(num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward through Tutel MoE.

        hidden_states: (batch, seq_len, d_model)
        Tutel expects shape (num_tokens, d_model), so we flatten batch/seq.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        output, aux_loss = self.moe_layer(x)
        self.load_balancing_loss = aux_loss

        output = self.dropout(output)
        output = output.reshape(batch_size, seq_len, hidden_dim)
        return output

    def get_load_balancing_loss(self):
        """Return Tutel's aux loss and a dummy per-expert usage vector.

        The first value is the aux (load balancing) loss produced by Tutel.
        The second is kept for API compatibility with the existing code.
        """
        return self.load_balancing_loss, self.total_fi


class FairscaleMoEBlock(nn.Module):
    """GShard-style Top-2 MoE block with memory-efficient sparse dispatch.

    Implements the routing algorithm from the GShard paper (identical to
    fairscale's Top2Gate), but dispatches tokens using a sparse expert loop
    instead of the capacity-buffer / AllToAll approach used by
    fairscale.nn.moe.MOELayer.

    Key differences vs. QwenStyleSparseMoEBlock:
      - 2nd expert is chosen via Gumbel-max trick (exploration / diversity)
        rather than plain top-2 of softmax.
      - l_aux = mean(me * ce)  (GShard / fairscale formula)
        where me = mean({gate prob}_e), ce = mean({top-1 dispatch freq}_e).
      - Routing decisions are stored as ``_last_selected_experts`` and
        ``_last_routing_weights`` after every forward pass, ready for
        routing-replay downstream use.

    Memory: O(N * E) like QwenStyle — no capacity buffer, no torch.distributed.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int = 2048,
        num_experts: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
        load_balancing_alpha: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = 2  # GShard is always top-2
        self.alpha = load_balancing_alpha

        # Gate: same architecture as fairscale Top2Gate.wg
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

        # Activation
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()

        # Experts (FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, intermediate_size, bias=False),
                act_fn,
                nn.Linear(intermediate_size, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)

        # Cached routing state (updated every forward)
        self.load_balancing_loss: torch.Tensor = torch.tensor(0.0)
        self.register_buffer("total_fi", torch.zeros(num_experts, dtype=torch.float32))
        # Routing replay buffers (detached, no grad)
        self._last_selected_experts: Optional[torch.Tensor] = None
        self._last_routing_weights: Optional[torch.Tensor] = None

    def _gshard_top2_routing(
        self, x: torch.Tensor
    ):
        """GShard Top-2 routing with Gumbel noise for the 2nd expert.

        Args:
            x: (N, d_model) — flattened token embeddings.

        Returns:
            routing_weights : (N, 2)  normalized weights (cast to x.dtype)
            selected_experts: (N, 2)  expert indices  [top-1, top-2]
            l_aux           : scalar  load-balancing loss
        """
        logits = self.gate(x)                                   # (N, E)
        gates  = F.softmax(logits, dim=-1, dtype=torch.float32) # (N, E) always fp32

        # ---- 1st expert: plain argmax ----
        indices1 = torch.argmax(gates, dim=-1)                  # (N,)
        mask1    = F.one_hot(indices1, num_classes=self.num_experts).float()  # (N, E)

        # ---- 2nd expert: Gumbel-max on remaining experts (GShard trick) ----
        gumbel = -torch.log(
            -torch.log(torch.rand_like(logits).clamp(min=1e-20)) + 1e-20
        )
        logits_except1 = (logits + gumbel).masked_fill(mask1.bool(), float("-inf"))
        indices2 = torch.argmax(logits_except1, dim=-1)         # (N,)
        mask2    = F.one_hot(indices2, num_classes=self.num_experts).float()  # (N, E)

        # ---- Normalize gate weights for the chosen pair ----
        gates1 = (gates * mask1).sum(dim=-1)                    # (N,)
        gates2 = (gates * mask2).sum(dim=-1)                    # (N,)
        denom  = (gates1 + gates2).clamp(min=torch.finfo(torch.float32).eps)
        gates1 = gates1 / denom
        gates2 = gates2 / denom

        routing_weights  = torch.stack([gates1, gates2], dim=-1)         # (N, 2)
        selected_experts = torch.stack([indices1, indices2], dim=-1)     # (N, 2)

        # ---- GShard l_aux = mean(me * ce) ----
        me    = gates.mean(dim=0)    # (E,) mean gate probability
        ce    = mask1.mean(dim=0)    # (E,) mean top-1 dispatch frequency
        l_aux = self.alpha * torch.mean(me * ce)

        return routing_weights.to(x.dtype), selected_experts, l_aux

    def forward(
        self,
        hidden_states: torch.Tensor,
        external_routing: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states   : (batch, seq_len, d_model)
            external_routing: optional ``(routing_weights, selected_experts)``
                              tuple for routing replay.  Both tensors must have
                              the same first two dims as the flattened batch.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)   # (N, d_model)
        N = x.shape[0]

        # ---- Routing ----
        if external_routing is not None:
            routing_weights, selected_experts = external_routing
            # Recompute l_aux from current gate so gradients flow correctly
            logits = self.gate(x)
            gates  = F.softmax(logits, dim=-1, dtype=torch.float32)
            indices1 = selected_experts[:, 0]
            mask1 = F.one_hot(indices1, num_classes=self.num_experts).float()
            me    = gates.mean(dim=0)
            ce    = mask1.mean(dim=0)
            self.load_balancing_loss = self.alpha * torch.mean(me * ce)
        else:
            routing_weights, selected_experts, l_aux = self._gshard_top2_routing(x)
            self.load_balancing_loss = l_aux

        # Cache routing for replay (always detached)
        self._last_selected_experts = selected_experts.detach()
        self._last_routing_weights  = routing_weights.detach()

        # Update per-expert usage fraction (top-1 based, no grad)
        with torch.no_grad():
            fi = torch.zeros(self.num_experts, dtype=torch.float32, device=x.device)
            fi.scatter_add_(
                0, selected_experts[:, 0],
                torch.ones(N, dtype=torch.float32, device=x.device)
            )
            self.total_fi = fi / N

        # ---- Sparse dispatch (memory-efficient, same as QwenStyle) ----
        final_hidden_states = torch.zeros_like(x)
        flat_experts = selected_experts.view(-1)                # (N*2,)
        flat_weights = routing_weights.view(-1)                 # (N*2,)
        token_indices = (
            torch.arange(N, device=x.device)
            .unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        )

        for expert_idx in range(self.num_experts):
            mask = (flat_experts == expert_idx)
            if not mask.any():
                continue
            expert_tokens  = token_indices[mask]
            expert_weights = flat_weights[mask]
            expert_out     = self.experts[expert_idx](x[expert_tokens])
            final_hidden_states.index_add_(
                0, expert_tokens, expert_out * expert_weights.unsqueeze(-1)
            )

        final_hidden_states = self.dropout(final_hidden_states)
        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

    def get_load_balancing_loss(self):
        """Return GShard l_aux and per-expert top-1 usage fraction."""
        return self.load_balancing_loss, self.total_fi


class FairscaleMoEBlock_RL(FairscaleMoEBlock):
    """R3 (Rollout Routing Replay) variant of FairscaleMoEBlock.

    核心改動：訓練時專家選擇不再由當前 logits 動態決定，
    而是固定使用推理階段記錄的路由掩碼 ``I_infer``。
    門控權重則對訓練時的 logits 做 **masked softmax**，
    只在 ``I_infer`` 選中的專家上歸一化，因此梯度仍可流回 router。

    計算流程：
      1. s_train = x_train @ W_r           (router logits，不變)
      2. I_infer = inference recorded mask  (專家選擇，來自推理)
      3. g_replay_i = I_infer_i * exp(s_train_i)
                      / sum_j I_infer_j * exp(s_train_j)
         → masked softmax over selected experts
      4. y_replay = sum_i g_replay_i * E_i(x_train)

    使用方式（策略 C — Detached Single-Step）：
      1. 外部先呼叫 ``model.diffmodel(...)`` (no_grad) 一次
         → 所有 block 的 ``_last_selected_experts`` 被記錄
      2. 呼叫 ``block.set_pending_routing(selected_experts)``
      3. 再次 ``model.diffmodel(...)`` (有 grad)
         → block 自動消費 ``_pending_routing``，以 R3 模式 forward
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_routing: Optional[torch.Tensor] = None

    def set_pending_routing(self, selected_experts: torch.Tensor):
        """設定下次 forward 要 replay 的路由。

        Args:
            selected_experts: (N, 2) int64 tensor，detached。
        """
        self._pending_routing = selected_experts

    def clear_pending_routing(self):
        """清除待用路由（安全用，forward 後也會自動清）。"""
        self._pending_routing = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        external_routing: Optional[tuple] = None,
    ) -> torch.Tensor:
        """R3 forward pass.

        Args:
            hidden_states   : (batch, seq_len, d_model)
            external_routing: optional ``(routing_weights_ignored, selected_experts_infer)``
                              where ``selected_experts_infer`` is (N, 2) expert
                              indices recorded during inference.
                              ``routing_weights_ignored`` is present for API
                              compatibility but **not used** — weights are
                              recomputed via masked softmax on training logits.
        """
        # ---- Consume _pending_routing (Strategy C one-shot injection) ----
        if external_routing is None and self._pending_routing is not None:
            external_routing = (None, self._pending_routing)
            self._pending_routing = None  # consumed

        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)  # (N, d_model)
        N = x.shape[0]

        if external_routing is not None:
            # ============================================================
            # R3 Mode: Routing Replay
            # ============================================================
            _, selected_experts_infer = external_routing  # (N, 2)

            # Step 1 — Router logits from training input (梯度來源)
            logits = self.gate(x)  # (N, E)

            # Step 2 — Expert selection fixed from inference
            selected_experts = selected_experts_infer  # (N, 2)

            # Step 3 — Masked softmax: g_replay
            #   build mask (N, E) marking the 2 selected experts per token
            infer_mask = torch.zeros(
                N, self.num_experts, device=x.device, dtype=torch.bool
            )
            infer_mask.scatter_(1, selected_experts, True)

            #   mask‑fill non‑selected experts to −inf, then softmax
            masked_logits = logits.masked_fill(~infer_mask, float("-inf"))
            gates_replay = F.softmax(masked_logits, dim=-1, dtype=torch.float32)  # (N, E)

            #   gather weights corresponding to the 2 selected experts
            routing_weights = torch.gather(gates_replay, 1, selected_experts)  # (N, 2)
            routing_weights = routing_weights.to(x.dtype)

            # l_aux: still use full softmax for load‑balancing signal
            full_gates = F.softmax(logits, dim=-1, dtype=torch.float32)
            indices1 = selected_experts[:, 0]
            mask1 = F.one_hot(indices1, num_classes=self.num_experts).float()
            me = full_gates.mean(dim=0)
            ce = mask1.mean(dim=0)
            self.load_balancing_loss = self.alpha * torch.mean(me * ce)
        else:
            # ============================================================
            # Normal / Inference Mode: identical to parent (GShard Top-2)
            # ============================================================
            routing_weights, selected_experts, l_aux = self._gshard_top2_routing(x)
            self.load_balancing_loss = l_aux

        # Cache routing for subsequent replay
        self._last_selected_experts = selected_experts.detach()
        self._last_routing_weights = routing_weights.detach()

        # Update per-expert usage fraction (top-1 based, no grad)
        with torch.no_grad():
            fi = torch.zeros(self.num_experts, dtype=torch.float32, device=x.device)
            fi.scatter_add_(
                0,
                selected_experts[:, 0],
                torch.ones(N, dtype=torch.float32, device=x.device),
            )
            self.total_fi = fi / N

        # ---- Sparse dispatch (same as parent) ----
        final_hidden_states = torch.zeros_like(x)
        flat_experts = selected_experts.view(-1)   # (N*2,)
        flat_weights = routing_weights.view(-1)    # (N*2,)
        token_indices = (
            torch.arange(N, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )

        for expert_idx in range(self.num_experts):
            emask = flat_experts == expert_idx
            if not emask.any():
                continue
            expert_tokens = token_indices[emask]
            expert_weights = flat_weights[emask]
            expert_out = self.experts[expert_idx](x[expert_tokens])
            final_hidden_states.index_add_(
                0, expert_tokens, expert_out * expert_weights.unsqueeze(-1)
            )

        final_hidden_states = self.dropout(final_hidden_states)
        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)