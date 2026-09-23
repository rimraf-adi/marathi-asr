"""
Sparse Upcycling Engine:
Clones dense Feed-Forward Network (FFN 2) weights from Pre-trained Conformer checkpoint
into the Combining Shared FFN Trunk and all 3 Regional Dialect Experts (D1, D2, D4).
Guarantees mathematically identical outputs at Step 0 of MoE Training.
"""

import os
from typing import Dict, Any, List
import torch
import torch.nn as nn

from moe.moe_layer import SparseMoELayer
from model.asr_model import StreamingASRModel


def upcycle_conformer_to_moe(
    pretrained_checkpoint_path: str,
    num_experts: int = 3,
    moe_layers: List[int] = [4, 5, 6, 7, 8, 9, 10, 11],  # Conformer blocks 5 to 12 (0-indexed)
    device: torch.device = torch.device("cpu"),
) -> StreamingASRModel:
    """
    Upcycles a dense Conformer into a 3-Dialect MoE Conformer.
    Args:
        pretrained_checkpoint_path: Path to conformer_pretrain_final.pt
        num_experts: Number of dialect experts (3: Malvani, Ahirani, Varhadi)
        moe_layers: Conformer block indices to convert to MoE (default: layers 5-12)
        device: Target device
    Returns:
        model: StreamingASRModel with Conformer blocks 5-12 upcycled to SparseMoELayer
    """
    print(f"\n[Upcycling] Loading Pre-trained checkpoint: {pretrained_checkpoint_path}")
    checkpoint = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Extract backbone state_dict if wrapped in JointMultiExitASR
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_state[k.replace("backbone.", "")] = v
        else:
            backbone_state[k] = v

    # Instantiate base model
    model = StreamingASRModel(
        feat_dim=80,
        d_model=256,
        num_layers=12,
        n_heads=4,
        conv_kernel_size=31,
        ffn_expansion=4,
        dropout=0.1,
        exit_layers=[4, 8, 12],
        enable_reconstruction_head=False,
        vocab_size=105,
    ).to(device)

    # Load initial dense weights into the entire model
    model.load_state_dict(backbone_state, strict=False)

    print(f"[Upcycling] Converting Conformer blocks {moe_layers} to 3-Dialect MoE...")

    for layer_idx in moe_layers:
        block = model.encoder.layers[layer_idx]
        original_ffn2 = block.ffn2

        # Create SparseMoELayer
        moe_module = SparseMoELayer(
            d_model=256,
            expansion_factor=4,
            num_experts=num_experts,
            dropout=0.1,
        ).to(device)

        # 1. Clone weights into Combining Shared FFN Trunk (Standard Marathi D3)
        moe_module.combining_ffn.layer_norm.weight.data.copy_(original_ffn2.layer_norm.weight.data)
        moe_module.combining_ffn.layer_norm.bias.data.copy_(original_ffn2.layer_norm.bias.data)
        moe_module.combining_ffn.linear1.weight.data.copy_(original_ffn2.linear1.weight.data)
        moe_module.combining_ffn.linear1.bias.data.copy_(original_ffn2.linear1.bias.data)
        moe_module.combining_ffn.linear2.weight.data.copy_(original_ffn2.linear2.weight.data)
        moe_module.combining_ffn.linear2.bias.data.copy_(original_ffn2.linear2.bias.data)

        # 2. Clone identical weights into ALL 3 Regional Dialect Experts (D1, D2, D4)
        for expert_id in range(num_experts):
            expert = moe_module.experts[expert_id]
            expert.layer_norm.weight.data.copy_(original_ffn2.layer_norm.weight.data)
            expert.layer_norm.bias.data.copy_(original_ffn2.layer_norm.bias.data)
            expert.linear1.weight.data.copy_(original_ffn2.linear1.weight.data)
            expert.linear1.bias.data.copy_(original_ffn2.linear1.bias.data)
            expert.linear2.weight.data.copy_(original_ffn2.linear2.weight.data)
            expert.linear2.bias.data.copy_(original_ffn2.linear2.bias.data)

        # Replace ffn2 with moe_module in the Conformer block
        block.ffn2 = moe_module

    print(f"[Upcycling] Successfully upcycled {len(moe_layers)} Conformer blocks to 3-Dialect MoE!")
    return model


def load_moe_model(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    num_experts: int = 3,
    moe_layers: List[int] = [4, 5, 6, 7, 8, 9, 10, 11],
) -> StreamingASRModel:
    """
    Directly instantiates a 3-Dialect MoE Conformer and loads trained weights.
    """
    model = StreamingASRModel(
        feat_dim=80,
        d_model=256,
        num_layers=12,
        n_heads=4,
        conv_kernel_size=31,
        ffn_expansion=4,
        dropout=0.1,
        exit_layers=[4, 8, 12],
        enable_reconstruction_head=False,
        vocab_size=105,
    ).to(device)

    for layer_idx in moe_layers:
        model.encoder.layers[layer_idx].ffn2 = SparseMoELayer(
            d_model=256,
            expansion_factor=4,
            num_experts=num_experts,
            dropout=0.1,
        ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model

