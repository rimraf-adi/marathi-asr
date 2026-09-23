"""
Reconstruction and CTC Heads for the Conformer Architecture.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class MaskedReconstructionHead(nn.Module):
    """
    Reconstructs the original 80-dim log-mel spectrogram frames from Conformer representations.
    Because the Conformer encoder applies 4x subsampling (Conv2dSubsampling4), each encoder step
    corresponds to 4 original spectrogram frames.
    This head projects d_model -> (4 * feat_dim) and reshapes back to the original time resolution.
    """

    def __init__(self, d_model: int = 256, feat_dim: int = 80, hidden_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.subsample_rate = 4

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.subsample_rate * feat_dim),
        )

    def forward(self, encoder_output: torch.Tensor, target_time_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, time // 4, d_model)
            target_time_len: Optional original time length to crop or pad to exact match
        Returns:
            reconstructed_spec: (batch, time, feat_dim)
        """
        b, t_sub, _ = encoder_output.size()
        out = self.net(encoder_output)  # (batch, time // 4, 4 * feat_dim)
        out = out.contiguous().view(b, t_sub * self.subsample_rate, self.feat_dim)

        if target_time_len is not None:
            if out.size(1) > target_time_len:
                out = out[:, :target_time_len, :]
            elif out.size(1) < target_time_len:
                diff = target_time_len - out.size(1)
                out = nn.functional.pad(out, (0, 0, 0, diff))

        return out


class CTCHead(nn.Module):
    """Linear CTC projection head from encoder hidden dimension to vocabulary."""

    def __init__(self, d_model: int = 256, vocab_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time // 4, d_model)
        Returns:
            log_probs: (batch, time // 4, vocab_size)
        """
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.proj(x)
        return torch.log_softmax(logits, dim=-1)


class MultiExitCTCHeads(nn.Module):
    """
    Multi-Exit CTC heads tapped at specified Conformer layers (e.g. layers 4, 8, 12).
    """

    def __init__(self, d_model: int = 256, vocab_size: int = 128, exit_layers: List[int] = [4, 8, 12]):
        super().__init__()
        self.exit_layers = exit_layers
        self.heads = nn.ModuleDict({
            f"exit_{layer}": CTCHead(d_model=d_model, vocab_size=vocab_size)
            for layer in exit_layers
        })

    def forward(self, exit_hiddens: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Args:
            exit_hiddens: Dict mapping layer index -> hidden tensor (batch, time, d_model)
        Returns:
            exit_log_probs: Dict mapping layer index -> log probabilities (batch, time, vocab)
        """
        results = {}
        for layer, hidden in exit_hiddens.items():
            key = f"exit_{layer}"
            if key in self.heads:
                results[layer] = self.heads[key](hidden)
        return results
