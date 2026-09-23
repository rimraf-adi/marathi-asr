"""
Academic & Thesis Metrics Tracker for Speech Self-Supervised Pretraining.
Computes detailed reconstruction, representation collapse, and hardware profiling metrics.
Exports directly to:
  - Step-level JSON Lines (metrics.jsonl)
  - Tabular CSV (summary_table.csv)
  - Formatted LaTeX Tables for thesis/paper inclusion (latex_table_pretrain.tex)
  - Visual high-resolution spectrogram plots (plots/step_XXXX.png)
"""

import os
import json
import math
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


class PretrainMetricsTracker:
    """
    Rigorously tracks and computes all quantitative and qualitative metrics for pretraining.
    """

    def __init__(self, log_dir: str = "runs/pretrain_experiment"):
        self.log_dir = log_dir
        self.plots_dir = os.path.join(log_dir, "plots")
        self.checkpoints_dir = os.path.join(log_dir, "checkpoints")

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self.csv_path = os.path.join(log_dir, "summary_table.csv")
        self.latex_path = os.path.join(log_dir, "latex_table_pretrain.tex")

        # Rolling accumulators for windowed statistics
        self.history: List[Dict[str, float]] = []

    # =========================================================================
    # 1. CORE RECONSTRUCTION METRICS
    # =========================================================================
    @staticmethod
    def compute_reconstruction_metrics(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Args:
            original: (B, T, F) ground truth spectrogram
            reconstructed: (B, T, F) predicted spectrogram
            mask: (B, T) boolean tensor (True = masked)
        Returns:
            Dictionary of mathematical reconstruction metrics
        """
        with torch.no_grad():
            b, t, f = original.size()
            mask_3d = mask.unsqueeze(-1).expand_as(original)  # (B, T, F)
            unmask_3d = ~mask_3d

            # Masked metrics
            if mask_3d.sum() > 0:
                masked_orig = original[mask_3d]
                masked_recon = reconstructed[mask_3d]

                masked_l1 = torch.abs(masked_orig - masked_recon).mean().item()
                masked_l2 = torch.mean((masked_orig - masked_recon) ** 2).item()

                # Reconstruction SNR in dB: 10 * log10( sum(S_true^2) / sum((S_true - S_pred)^2) )
                signal_power = torch.sum(masked_orig ** 2)
                noise_power = torch.sum((masked_orig - masked_recon) ** 2) + 1e-8
                snr_db = (10.0 * torch.log10(signal_power / noise_power)).item()

                # Spectral Convergence: ||S_true - S_pred||_F / ||S_true||_F
                spec_conv = (torch.norm(masked_orig - masked_recon) / (torch.norm(masked_orig) + 1e-8)).item()

                # Cosine similarity across frequency bins
                orig_flat = original.contiguous().reshape(b * t, f)[mask.reshape(-1)]
                recon_flat = reconstructed.contiguous().reshape(b * t, f)[mask.reshape(-1)]
                cos_sim = torch.cosine_similarity(orig_flat, recon_flat, dim=-1).mean().item()
            else:
                masked_l1 = masked_l2 = snr_db = spec_conv = cos_sim = 0.0

            # Unmasked metrics (checking model preserves unmasked regions)
            if unmask_3d.sum() > 0:
                unmasked_l1 = torch.abs(original[unmask_3d] - reconstructed[unmask_3d]).mean().item()
                unmasked_l2 = torch.mean((original[unmask_3d] - reconstructed[unmask_3d]) ** 2).item()
            else:
                unmasked_l1 = unmasked_l2 = 0.0

            return {
                "loss_masked_l1": masked_l1,
                "loss_masked_l2_mse": masked_l2,
                "loss_unmasked_l1": unmasked_l1,
                "loss_unmasked_l2_mse": unmasked_l2,
                "reconstruction_snr_db": snr_db,
                "spectral_convergence": spec_conv,
                "masked_cosine_similarity": cos_sim,
                "mask_ratio": mask.float().mean().item(),
            }

    # =========================================================================
    # 2. REPRESENTATION & COLLAPSE METRICS (PHASE 3 INTRINSIC GATE)
    # =========================================================================
    @staticmethod
    def compute_representation_metrics(hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        Calculates geometric and rank properties of the encoder representations.
        Detects dimensional collapse (all vectors lying on a low-rank subspace)
        or constant collapse (all vectors becoming identical).

        Args:
            hidden_states: (B, T_sub, D)
        """
        with torch.no_grad():
            b, t, d = hidden_states.size()
            # Flatten to (B * T_sub, D)
            flat = hidden_states.reshape(b * t, d).float()

            # Center the representations
            centered = flat - flat.mean(dim=0, keepdim=True)

            # SVD to compute singular values
            try:
                s = torch.linalg.svdvals(centered)  # (min(B*T, D),)
                # Normalize singular values into probability distribution
                s_sum = s.sum() + 1e-10
                p = s / s_sum
                # Effective rank = exp( - sum(p * log(p)) )
                entropy = -(p * torch.log(p + 1e-12)).sum()
                effective_rank = torch.exp(entropy).item()
            except Exception:
                effective_rank = 0.0

            # Feature variance across time
            feat_var = flat.var(dim=0).mean().item()

            # Average pairwise cosine similarity between consecutive time frames
            if t > 1:
                consec_cos = torch.cosine_similarity(
                    hidden_states[:, :-1, :], hidden_states[:, 1:, :], dim=-1
                ).mean().item()
            else:
                consec_cos = 1.0

            return {
                "effective_rank": effective_rank,
                "max_possible_rank": float(min(b * t, d)),
                "rank_utilization_pct": (effective_rank / float(min(b * t, d))) * 100.0 if d > 0 else 0.0,
                "representation_variance": feat_var,
                "temporal_cosine_sim": consec_cos,
                "hidden_norm_mean": flat.norm(dim=-1).mean().item(),
            }

    # =========================================================================
    # 3. LOGGING & ACCUMULATION
    # =========================================================================
    def log_step(
        self,
        step: int,
        epoch: int,
        loss_dict: Dict[str, float],
        rep_dict: Dict[str, float],
        perf_dict: Dict[str, float],
        lr: float,
    ):
        """Append record to jsonl file and rolling history."""
        record = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            "learning_rate": lr,
            **loss_dict,
            **rep_dict,
            **perf_dict,
        }

        self.history.append(record)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # =========================================================================
    # 4. QUALITATIVE SPECTROGRAM VISUALIZATION
    # =========================================================================
    def save_spectrogram_plot(
        self,
        step: int,
        original: torch.Tensor,
        masked: torch.Tensor,
        reconstructed: torch.Tensor,
        sample_idx: int = 0,
    ):
        """
        Saves a 4-panel publication-ready comparison figure:
          1. Ground Truth Log-Mel Spectrogram
          2. Masked Spectrogram (Model Input)
          3. Reconstructed Spectrogram (Model Prediction)
          4. Absolute Error Residual Map (|True - Pred|)
        """
        orig = original[sample_idx].detach().cpu().numpy().T       # (Freq, Time)
        mask_spec = masked[sample_idx].detach().cpu().numpy().T
        recon = reconstructed[sample_idx].detach().cpu().numpy().T
        residual = np.abs(orig - recon)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        vmin, vmax = -6.0, 3.0  # Normalized log-mel range

        im0 = axes[0].imshow(orig, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Step {step}: Ground Truth Log-Mel Spectrogram", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Mel Bin (80)")
        fig.colorbar(im0, ax=axes[0], pad=0.01)

        im1 = axes[1].imshow(mask_spec, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Masked Spectrogram (Encoder Input with Spans Zeroed)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Mel Bin (80)")
        fig.colorbar(im1, ax=axes[1], pad=0.01)

        im2 = axes[2].imshow(recon, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[2].set_title("Reconstructed Spectrogram (Model Output)", fontsize=11, fontweight="bold")
        axes[2].set_ylabel("Mel Bin (80)")
        fig.colorbar(im2, ax=axes[2], pad=0.01)

        im3 = axes[3].imshow(residual, origin="lower", aspect="auto", cmap="inferno", vmin=0.0, vmax=2.5)
        axes[3].set_title("Reconstruction Error Residual (|Ground Truth - Reconstructed|)", fontsize=11, fontweight="bold")
        axes[3].set_ylabel("Mel Bin (80)")
        axes[3].set_xlabel("Time Frame (10ms steps)")
        fig.colorbar(im3, ax=axes[3], pad=0.01)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"recon_step_{step:06d}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # =========================================================================
    # 5. EXPORT TABLES FOR THESIS & PAPER
    # =========================================================================
    def export_summary_and_latex(self, last_n_steps: int = 100):
        """
        Averages metrics over the most recent steps and creates:
          1. summary_table.csv
          2. latex_table_pretrain.tex ready for thesis / paper!
        """
        if not self.history:
            return

        window = self.history[-last_n_steps:]

        def get_stats(key):
            vals = [r[key] for r in window if key in r]
            if not vals:
                return 0.0, 0.0
            return float(np.mean(vals)), float(np.std(vals))

        metrics_to_report = [
            ("loss_masked_l1", "Masked L1 Loss"),
            ("loss_masked_l2_mse", "Masked L2 MSE Loss"),
            ("reconstruction_snr_db", "Reconstruction SNR (dB)"),
            ("spectral_convergence", "Spectral Convergence"),
            ("masked_cosine_similarity", "Masked Cosine Similarity"),
            ("effective_rank", "Effective Hidden Rank"),
            ("rank_utilization_pct", "Rank Utilization (%)"),
            ("representation_variance", "Feature Variance"),
            ("throughput_audio_sec_per_sec", "Throughput (Audio s/s)"),
            ("gpu_vram_peak_gb", "Peak VRAM (GB)"),
        ]

        # 1. Write CSV
        import csv
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric Name", "LaTeX Symbol / Description", "Mean", "StdDev"])
            for key, desc in metrics_to_report:
                mean_val, std_val = get_stats(key)
                writer.writerow([key, desc, f"{mean_val:.4f}", f"{std_val:.4f}"])

        # 2. Write LaTeX Table for Paper / Thesis
        latex_str = r"""\begin{table}[htbp]
\centering
\caption{Self-Supervised Pretraining Metrics on Marathi \& Regional Dialects (Conformer Backbone)}
\label{tab:marathi_ssl_pretrain_metrics}
\begin{tabular}{lrr}
\toprule
\textbf{Evaluation Metric} & \textbf{Mean} & \textbf{Std. Dev.} \\
\midrule
"""
        for key, desc in metrics_to_report:
            mean_val, std_val = get_stats(key)
            latex_str += f"{desc:35s} & {mean_val:10.4f} & $\\pm$ {std_val:.4f} \\\\\n"

        latex_str += r"""\bottomrule
\end{tabular}
\end{table}
"""
        with open(self.latex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
