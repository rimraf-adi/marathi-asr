"""
Spectrogram Masking modules for Self-Supervised Learning and Data Augmentation.
Includes:
  1. SpecAugment (Frequency & Time masking)
  2. Span-based Masking for SSL Pretraining (with short vs. long span configuration for intrinsic evaluation)
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn


class SpanMasker(nn.Module):
    """
    Span-based temporal masking for Self-Supervised Pretraining.
    Masks contiguous spans of time frames in the log-mel spectrogram.

    Args:
        mask_prob: Proportion of total time frames to mask (e.g., 0.40 - 0.50).
        mean_span_length: Average length of contiguous masked frames (e.g., 10-15 frames = 100-150ms).
        min_span_length: Minimum span length.
        mask_value: Fill value for masked positions (default: 0.0, or learnable/gaussian noise).
    """

    def __init__(
        self,
        mask_prob: float = 0.40,
        mean_span_length: int = 10,
        min_span_length: int = 2,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mean_span_length = mean_span_length
        self.min_span_length = min_span_length
        self.mask_value = mask_value

    def forward(
        self,
        x: torch.Tensor,
        custom_span_length: Optional[int] = None,
        custom_mask_prob: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input spectrogram tensor of shape (batch, time, freq)
            custom_span_length: Override span length (e.g. 4 for short-mask test, 20 for long-mask test)
            custom_mask_prob: Override mask probability
        Returns:
            masked_x: Spectrogram with spans zeroed out / filled
            mask: Boolean tensor of shape (batch, time) where True = masked
        """
        b, t, f = x.size()
        span_len = custom_span_length or self.mean_span_length
        prob = custom_mask_prob or self.mask_prob

        # Number of frames to mask per sequence
        num_masked_frames = int(t * prob)
        if num_masked_frames == 0 or span_len == 0:
            return x, torch.zeros((b, t), dtype=torch.bool, device=x.device)

        num_spans = max(1, num_masked_frames // span_len)
        max_start = max(1, t - span_len + 1)

        # Vectorized random start positions (batch, num_spans)
        starts = torch.randint(0, max_start, (b, num_spans), device=x.device)

        # Span frame offsets: (1, 1, span_len)
        offsets = torch.arange(span_len, device=x.device).unsqueeze(0).unsqueeze(0)

        # Broadcasted indices: (batch, num_spans, span_len)
        masked_indices = (starts.unsqueeze(-1) + offsets).clamp(max=t - 1)

        # Create boolean mask on GPU via scatter
        mask = torch.zeros((b, t), dtype=torch.bool, device=x.device)
        mask.scatter_(dim=1, index=masked_indices.view(b, -1), value=True)

        # Apply mask: where mask is True, replace with mask_value
        masked_x = x.clone()
        masked_x[mask.unsqueeze(-1).expand_as(x)] = self.mask_value

        return masked_x, mask


class SpecAugment(nn.Module):
    """
    Standard SpecAugment for acoustic regularization:
      - Random frequency channel masking
      - Random time frame masking
    """

    def __init__(
        self,
        freq_mask_max: int = 27,
        time_mask_max: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 5,
    ):
        super().__init__()
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, freq)
        """
        if not self.training:
            return x

        b, t, f = x.size()
        x = x.clone()

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f_len = torch.randint(0, self.freq_mask_max + 1, (1,)).item()
            f_start = torch.randint(0, max(1, f - f_len + 1), (1,)).item()
            x[:, :, f_start : f_start + f_len] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t_len = torch.randint(0, self.time_mask_max + 1, (1,)).item()
            t_start = torch.randint(0, max(1, t - t_len + 1), (1,)).item()
            x[:, t_start : t_start + t_len, :] = 0.0

        return x
