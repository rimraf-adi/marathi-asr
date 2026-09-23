"""
Stage 4: 3-Dialect Mixture-of-Experts (MoE) Layer with Combining Shared Trunk.
Implements the Shared-Trunk + Routed-Experts architecture:
  y = FFN_combining(x) + g_k * Expert_k(x)
Where:
  - FFN_combining: Always-active standard Marathi anchor (D3)
  - 3 Dialect Experts: D1 (Malvani), D2 (Ahirani), D4 (Varhadi)
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conformer import FeedForwardModule


class SparseMoELayer(nn.Module):
    """
    3-Dialect Mixture-of-Experts with Combining Shared FFN Trunk.
    Args:
        d_model: Hidden dimension (default: 256)
        expansion_factor: Expansion factor in FFN (default: 4)
        num_experts: Number of dialect experts (default: 3)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 256,
        expansion_factor: int = 4,
        num_experts: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        # Combining Shared FFN Trunk (Always-active standard Marathi baseline)
        self.combining_ffn = FeedForwardModule(d_model, expansion_factor, dropout)

        # 3 Dedicated Dialect Experts
        # Expert 0: D1 (Malvani / Konkan)
        # Expert 1: D2 (Ahirani / Khandesh)
        # Expert 2: D4 (Varhadi / Vidarbha)
        self.experts = nn.ModuleList([
            FeedForwardModule(d_model, expansion_factor, dropout)
            for _ in range(num_experts)
        ])

        # Lightweight Gating Router (maps frame hidden state to expert logits)
        self.router = nn.Linear(d_model, num_experts)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.router.bias)

    def forward(
        self,
        x: torch.Tensor,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: Input hidden states (batch, time, d_model)
            dialect_idx: Optional tensor of dialect indices (batch,) where:
                         0 = D1 (Malvani)
                         1 = D2 (Ahirani)
                         2 = D4 (Varhadi)
                         3 = D3 (Standard Marathi - uses shared trunk only)
            use_hard_routing: If True, routes strictly based on dialect_idx (Phase 4A)
        Returns:
            out: Combined hidden states (batch, time, d_model)
            aux_loss: Auxiliary load balancing loss
            routing_stats: Dictionary containing router distribution metrics
        """
        b, t, d = x.size()

        # 1. Always compute combining shared FFN trunk (Standard Marathi base)
        shared_out = self.combining_ffn(x)

        # 2. Router logits & probabilities across frames
        router_logits = self.router(x)  # (b, t, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (b, t, num_experts)

        top1_weights, top1_idx = torch.topk(router_probs, k=1, dim=-1)  # (b, t, 1)
        density = router_probs.mean(dim=(0, 1))  # (num_experts,)

        # Standard Switch Transformer load balancing loss: num_experts * sum(f_i * P_i)
        flat_top1_idx = top1_idx.view(-1)
        fractions = torch.bincount(flat_top1_idx, minlength=self.num_experts).float()
        fractions = fractions / max(1, fractions.sum().item())
        aux_loss = self.num_experts * torch.sum(fractions.detach() * density)

        if use_hard_routing and dialect_idx is not None:
            # Phase 4A: Metadata-guided routing
            expert_out = torch.zeros_like(x)
            for expert_id in range(self.num_experts):
                sample_mask = (dialect_idx == expert_id)
                if sample_mask.any():
                    expert_input = x[sample_mask]  # (B_sub, T, d)
                    expert_delta = self.experts[expert_id](expert_input) - expert_input
                    sub_weights = router_probs[sample_mask, :, expert_id:expert_id+1]
                    expert_out[sample_mask] = sub_weights * expert_delta
        else:
            # Phase 4B: Dynamic Soft/Top-1 routing
            flat_x = x.view(-1, d)
            flat_top1_weights = top1_weights.view(-1, 1)
            flat_expert_out = torch.zeros_like(flat_x)

            for expert_id in range(self.num_experts):
                mask = (flat_top1_idx == expert_id)
                if mask.any():
                    expert_inp = flat_x[mask]
                    expert_delta = self.experts[expert_id](expert_inp) - expert_inp
                    flat_expert_out[mask] = flat_top1_weights[mask] * expert_delta

            expert_out = flat_expert_out.view(b, t, d)

        # 3. Additive Fusion: combining trunk + routed dialect expert
        out = shared_out + expert_out

        self.current_aux_loss = aux_loss
        self.current_routing_stats = {
            f"expert_{i}_load": float(density[i].item())
            for i in range(self.num_experts)
        }

        return out
