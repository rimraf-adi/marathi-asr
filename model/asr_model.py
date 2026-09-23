"""
Unified Streaming Conformer ASR Model:
Integrates Log-Mel front-end, Conformer Encoder, SSL Reconstruction Head, and Multi-Exit CTC Heads.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torchaudio.transforms as T

from .conformer import ConformerEncoder
from .heads import MaskedReconstructionHead, MultiExitCTCHeads, CTCHead
from .masking import SpanMasker, SpecAugment


class LogMelSpectrogramFrontEnd(nn.Module):
    """
    Computes 80-dim log-mel filterbanks directly on GPU from raw 16kHz audio waveforms.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,       # 25 ms window
        hop_length: int = 160,  # 10 ms hop (100 frames/sec)
        n_mels: int = 80,
    ):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, num_samples) float tensor
        Returns:
            log_mel: (batch, time_frames, 80)
        """
        # mel_transform expects (batch, samples) -> (batch, n_mels, time)
        mel = self.mel_transform(waveforms)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        # Transpose to (batch, time, 80)
        return log_mel.transpose(1, 2)


class StreamingASRModel(nn.Module):
    """
    End-to-end model for both Self-Supervised Pretraining and Multi-Exit Downstream ASR.
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
        exit_layers: List[int] = [4, 8, 12],
        vocab_size: Optional[int] = None,
        enable_reconstruction_head: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.exit_layers = exit_layers

        # 1. Front-end feature extractor
        self.frontend = LogMelSpectrogramFrontEnd(n_mels=feat_dim)

        # 2. Masker for SSL pretraining
        self.span_masker = SpanMasker(mask_prob=0.40, mean_span_length=10)
        self.spec_augment = SpecAugment()

        # 3. Conformer Backbone
        self.encoder = ConformerEncoder(
            feat_dim=feat_dim,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            conv_kernel_size=conv_kernel_size,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            exit_layers=exit_layers,
        )

        # 4. Reconstruction Head (for Phase 2 SSL pretraining)
        if enable_reconstruction_head:
            self.reconstruction_head = MaskedReconstructionHead(
                d_model=d_model, feat_dim=feat_dim, hidden_dim=512
            )
        else:
            self.reconstruction_head = None

        # 5. Multi-Exit CTC Heads (for Stages 3 & 4)
        if vocab_size is not None:
            self.ctc_heads = MultiExitCTCHeads(
                d_model=d_model, vocab_size=vocab_size, exit_layers=exit_layers
            )
        else:
            self.ctc_heads = None

    def forward_pretrain(
        self,
        waveforms: torch.Tensor,
        custom_span_length: Optional[int] = None,
        custom_mask_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for Self-Supervised Masked Reconstruction Pretraining.
        Args:
            waveforms: (batch, num_samples) raw audio at 16kHz
        Returns:
            dict containing:
              - 'original_spec': Ground truth log-mel spectrogram (B, T, 80)
              - 'masked_spec': Spectrogram with masked spans (B, T, 80)
              - 'mask': Boolean mask tensor (B, T) where True = masked
              - 'reconstructed_spec': Predicted spectrogram (B, T, 80)
              - 'final_hidden': Encoder hidden states (B, T//4, d_model)
              - 'exit_hiddens': Dict of hidden states at exit layers
        """
        # 1. Compute ground truth spectrogram
        spec = self.frontend(waveforms)
        t_len = spec.size(1)

        # 2. Apply span masking
        masked_spec, mask = self.span_masker(
            spec,
            custom_span_length=custom_span_length,
            custom_mask_prob=custom_mask_prob,
        )

        # 3. Pass masked spectrogram through Conformer encoder
        enc_out = self.encoder(masked_spec, return_all_exits=True)
        final_hidden = enc_out["final_hidden"]

        # 4. Reconstruct spectrogram from final encoder representation
        reconstructed_spec = self.reconstruction_head(final_hidden, target_time_len=t_len)

        return {
            "original_spec": spec,
            "masked_spec": masked_spec,
            "mask": mask,
            "reconstructed_spec": reconstructed_spec,
            "final_hidden": final_hidden,
            "exit_hiddens": enc_out["exit_hiddens"],
        }

    def forward_ctc(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        dialect_idx: Optional[torch.Tensor] = None,
        use_hard_routing: bool = False,
        max_exit_layer: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for Causal Adaptation & Supervised CTC Fine-Tuning.
        Args:
            waveforms: (batch, num_samples) raw audio at 16kHz
            attention_mask: Optional precomputed dynamic chunk mask
            chunk_size: Optional dynamic chunk size in subsampled frames (e.g. 1, 4, 8, 16, -1)
            dialect_idx: Optional dialect indices for MoE routing
            use_hard_routing: If True, force hard routing in MoE layers
            max_exit_layer: If specified, halt computation after reaching this exit layer
        Returns:
            dict containing:
              - 'log_probs': (batch, T//4, vocab_size) at final layer
              - 'exit_log_probs': Dict[int, Tensor] mapping layer -> (batch, T//4, vocab_size)
              - 'output_lengths': (batch,) subsampled sequence lengths
              - 'final_hidden': (batch, T//4, d_model)
        """
        spec = self.frontend(waveforms)
        if self.training:
            spec = self.spec_augment(spec)
        enc_out = self.encoder(
            spec,
            mask=attention_mask,
            chunk_size=chunk_size,
            return_all_exits=True,
            dialect_idx=dialect_idx,
            use_hard_routing=use_hard_routing,
            max_exit_layer=max_exit_layer,
        )
        final_hidden = enc_out["final_hidden"]

        b, t, _ = final_hidden.size()
        output_lengths = torch.full((b,), t, dtype=torch.long, device=waveforms.device)

        exit_log_probs = {}
        final_log_probs = None

        if self.ctc_heads is not None:
            exit_log_probs = self.ctc_heads(enc_out["exit_hiddens"])
            final_layer_idx = max(enc_out["exit_hiddens"].keys())
            final_log_probs = exit_log_probs[final_layer_idx]

        return {
            "log_probs": final_log_probs,
            "exit_log_probs": exit_log_probs,
            "output_lengths": output_lengths,
            "final_hidden": final_hidden,
        }
