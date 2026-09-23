"""
Conformer: Convolution-augmented Transformer for Speech Recognition.
Reference: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech 2020.
Includes support for causal attention masking, dynamic chunking, and multi-exit extraction.
"""

import math
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv2dSubsampling4(nn.Module):
    """
    4x Convolutional Subsampling block.
    Reduces 100 fps spectrogram frames to 25 fps (40ms per frame) using two 2D strided convolutions.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 256, feat_dim: int = 80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        # Calculate output feature dimension after 2 strides of 2
        subsampled_feat_dim = feat_dim // 4
        self.out_proj = nn.Linear(out_dim * subsampled_feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, feat_dim)
        Returns:
            out: (batch, time // 4, out_dim)
        """
        # (batch, 1, time, feat_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (batch, out_dim, time // 4, feat_dim // 4)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        out = self.out_proj(x)
        return out


class FeedForwardModule(nn.Module):
    """
    Conformer Feed-Forward Module.
    Consists of LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout.
    Used in a Macaron-style half-step residual structure.
    """

    def __init__(self, d_model: int = 256, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()  # Swish
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    """
    Conformer Convolution Module.
    Order:
      LayerNorm
      -> Pointwise Conv1d (d_model -> 2 * d_model)
      -> Gated Linear Unit (GLU)
      -> 1D Depthwise Conv (kernel_size=31)
      -> BatchNorm1d
      -> Swish (SiLU)
      -> Pointwise Conv1d
      -> Dropout
    """

    def __init__(self, d_model: int = 256, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd for symmetric padding"
        self.padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model,
            bias=True,
        )
        self.norm = nn.GroupNorm(1, d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        # Conv1d operates on (batch, channel, time)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for relative positional attention.
    Encodes relative position directly into query-key dot products:
    <R_m q_m, R_n k_n> = <q_m, R_{n-m} k_n>
    """

    def __init__(self, d_k: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k shape: (b, n_heads, t, d_k)
        t = q.size(2)
        if t > self.cos_cached.size(0):
            self._build_cache(t + 512)
        cos = self.cos_cached[:t, :].to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:t, :].to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            half = x.shape[-1] // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with support for attention masks (causal, chunk, or padding)
    and Rotary Position Embeddings (RoPE) for relative positional awareness.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, time, d_model)
            mask: Optional attention mask (batch, 1, time, time) or (1, 1, time, time)
                  where True/1 indicates allowed and False/0 indicates masked.
        """
        residual = x
        x = self.layer_norm(x)
        b, t, _ = x.size()

        q = self.q_proj(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)

        # Apply Rotary Position Embedding to Queries and Keys
        q, k = self.rope(q, k)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # In FP16/AMP, -1e4 prevents softmax NaN underflow while setting attention to 0
            neg_inf = -1e4 if scores.dtype == torch.float16 else -1e9
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, neg_inf)
            else:
                scores = scores.masked_fill(mask == 0, neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (b, n_heads, t, d_k)
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out_proj(context)
        out = self.dropout(out)
        return residual + out


class ConformerBlock(nn.Module):
    """
    A single Conformer block comprising:
      1. Feed-Forward Module (half-step)
      2. Multi-Head Self-Attention
      3. Convolution Module
      4. Feed-Forward Module (half-step)
      5. Final LayerNorm
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
    ) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.self_attn(x, mask=mask)
        x = self.conv_module(x)
        if type(self.ffn2).__name__ == "SparseMoELayer":
            x = self.ffn2(x, dialect_idx=dialect_idx, use_hard_routing=use_hard_routing)
        else:
            x = self.ffn2(x)
        x = self.final_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """
    Full Conformer Acoustic Encoder with Multi-Exit taps.
    Args:
        feat_dim: Input audio feature dimension (default: 80 log-mel filterbanks)
        d_model: Encoder hidden dimension (default: 256)
        num_layers: Number of Conformer blocks (default: 12)
        n_heads: Attention heads (default: 4)
        conv_kernel_size: Convolution module kernel size (default: 31)
        ffn_expansion: Expansion factor in FFN (default: 4)
        dropout: Dropout rate (default: 0.1)
        exit_layers: Layers to extract intermediate representations (e.g. [4, 8, 12])
    """

    def __init__(
        self,
        feat_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 12,
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        exit_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.exit_layers = exit_layers or [num_layers]

        self.subsampling = Conv2dSubsampling4(in_channels=1, out_dim=d_model, feat_dim=feat_dim)
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        return_all_exits: bool = False,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
        max_exit_layer: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            x: Input spectrogram tensor of shape (batch, time_frames, 80)
            mask: Optional precomputed attention mask
            chunk_size: Optional dynamic chunk size in subsampled frames (e.g. 1, 4, 8, 16, -1)
            return_all_exits: If True, return representations at all specified exit_layers
            dialect_idx: Optional regional dialect indices for MoE routing
            use_hard_routing: If True, force hard dialect routing in MoE layers
            max_exit_layer: If specified, halt computation immediately after reaching this exit layer
        Returns:
            dict containing:
              - 'final_hidden': Tensor of shape (batch, subsampled_time, d_model)
              - 'exit_hiddens': Dict[int, Tensor] for each exit layer index (1-based)
        """
        x = self.subsampling(x)  # (batch, subsampled_time, d_model)
        x = self.pos_enc(x)
        b, t, _ = x.size()

        if chunk_size is not None:
            if chunk_size == -1 or chunk_size >= t:
                mask = None
            else:
                grid_i = torch.arange(t, device=x.device).unsqueeze(1)
                grid_j = torch.arange(t, device=x.device).unsqueeze(0)
                mask = (grid_j // chunk_size <= grid_i // chunk_size).unsqueeze(0).unsqueeze(0)

        exit_hiddens: Dict[int, torch.Tensor] = {}

        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask=mask, dialect_idx=dialect_idx, use_hard_routing=use_hard_routing)
            if idx in self.exit_layers:
                exit_hiddens[idx] = x
            if max_exit_layer is not None and idx >= max_exit_layer:
                break

        return {
            "final_hidden": x,
            "exit_hiddens": exit_hiddens,
        }
