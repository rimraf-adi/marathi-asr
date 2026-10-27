"""
Multi-Dialect Low-Rank Adaptation (LoRA) for Conformer ASR Models.
Allows parameter-efficient fine-tuning (PEFT) on distinct regional dialects
while keeping the 130M dense Conformer backbone completely frozen.
"""

import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDialectLoRALinear(nn.Module):
    """
    Multi-Dialect LoRA Linear Layer.
    Wraps an existing frozen nn.Linear and introduces K low-rank adapter paths:
    h = W_0 x + (alpha / r) * (x @ A_k^T) @ B_k^T
    where k in {0, ..., num_dialects - 1}.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        num_dialects: int = 3,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.num_dialects = num_dialects
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze base linear
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        # LoRA parameters for each dialect: A_k and B_k
        self.lora_A = nn.Parameter(torch.empty(num_dialects, r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(num_dialects, self.out_features, r, device=device, dtype=dtype))

        # Reset parameters: Kaiming uniform for A, zero for B (starts as exact identity)
        for d in range(num_dialects):
            nn.init.kaiming_uniform_(self.lora_A[d], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[d])

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.active_adapter: Optional[int] = None

    def forward(self, x: torch.Tensor, adapter_idx: Optional[int] = None) -> torch.Tensor:
        base_out = self.base_linear(x)
        idx = adapter_idx if adapter_idx is not None else self.active_adapter
        if idx is None or idx < 0 or idx >= self.num_dialects:
            return base_out

        # Compute LoRA forward: (x @ A_k^T) @ B_k^T * scaling
        x_drop = self.dropout(x)
        lora_h = F.linear(x_drop, self.lora_A[idx])  # (batch, time, r)
        lora_out = F.linear(lora_h, self.lora_B[idx]) * self.scaling  # (batch, time, out_features)
        return base_out + lora_out


def inject_dialect_lora(
    model: nn.Module,
    num_dialects: int = 3,
    r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    target_layers: str = "all",  # 'all' (Layers 1-12) or 'upper8' (Layers 5-12)
) -> nn.Module:
    """
    Injects MultiDialectLoRALinear into self-attention (q, v) and feed-forward modules.
    Freezes all non-LoRA parameters.
    """
    # 1. Freeze all parameters in the model first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Identify target layers
    encoder = getattr(model, "encoder", model)
    blocks = getattr(encoder, "layers", getattr(encoder, "blocks", getattr(encoder, "conformer_blocks", [])))
    if not blocks:
        raise ValueError("Could not find Conformer blocks/layers in model.encoder")

    num_layers = len(blocks)
    if target_layers == "upper8":
        start_idx = max(0, num_layers - 8)
    else:
        start_idx = 0

    injected_modules = 0
    for l_idx in range(start_idx, num_layers):
        block = blocks[l_idx]

        # Multi-Head Self-Attention projections (Q and V)
        if hasattr(block, "self_attn"):
            sa = block.self_attn
            if isinstance(sa.q_proj, nn.Linear):
                sa.q_proj = MultiDialectLoRALinear(sa.q_proj, num_dialects, r, lora_alpha, lora_dropout)
                injected_modules += 1
            if isinstance(sa.v_proj, nn.Linear):
                sa.v_proj = MultiDialectLoRALinear(sa.v_proj, num_dialects, r, lora_alpha, lora_dropout)
                injected_modules += 1

        # Feed-Forward Module 1
        if hasattr(block, "ffn1"):
            ffn = block.ffn1
            if isinstance(ffn.linear1, nn.Linear):
                ffn.linear1 = MultiDialectLoRALinear(ffn.linear1, num_dialects, r, lora_alpha, lora_dropout)
                injected_modules += 1
            if isinstance(ffn.linear2, nn.Linear):
                ffn.linear2 = MultiDialectLoRALinear(ffn.linear2, num_dialects, r, lora_alpha, lora_dropout)
                injected_modules += 1

        # Feed-Forward Module 2 (if standard linear)
        if hasattr(block, "ffn2") and not type(block.ffn2).__name__.startswith("SparseMoE"):
            ffn = block.ffn2
            if isinstance(ffn.linear1, nn.Linear):
                ffn.linear1 = MultiDialectLoRALinear(ffn.linear1, num_dialects, r, lora_alpha, lora_dropout)
                injected_modules += 1
            if isinstance(ffn.linear2, nn.Linear):
                ffn.linear2 = MultiDialectLoRALinear(ffn.linear2, num_dialects, r, lora_alpha, lora_dropout)
                injected_modules += 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_dialect_params = trainable_params // num_dialects

    print(f"\n[LoRA Injection] Target Layers: {target_layers} (Layers {start_idx+1} to {num_layers})")
    print(f"  Injected LoRA Linear Modules: {injected_modules}")
    print(f"  Total Model Parameters: {total_params:,}")
    print(f"  Total Trainable LoRA Parameters ({num_dialects} dialects): {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"  Trainable Parameters per Dialect Adapter: {per_dialect_params:,} ({per_dialect_params / total_params * 100:.2f}%)")

    return model


def set_active_lora_adapter(model: nn.Module, adapter_idx: Optional[int]):
    """Sets active dialect adapter across all MultiDialectLoRALinear layers in model."""
    for m in model.modules():
        if isinstance(m, MultiDialectLoRALinear):
            m.active_adapter = adapter_idx


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts only LoRA parameter tensors (lora_A, lora_B)."""
    return {k: v.cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]):
    """Loads LoRA parameters into wrapped model."""
    model.load_state_dict(lora_state, strict=False)
