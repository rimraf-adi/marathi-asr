"""
Dump all training metrics from task log and metrics.jsonl to structured CSV files for plotting.
Produces:
  1. pretrain_full_step_metrics.csv       (High-resolution step-by-step metrics for all steps)
  2. pretrain_smoothed_metrics.csv        (50-step rolling averages for clean thesis/paper plotting)
  3. pretrain_phase3_gates.csv            (Intrinsic gate evaluations extracted directly from task log)
  4. pretrain_tasklog_console_steps.csv   (Console checkpoints extracted directly from task log)
  5. pretrain_publication_plots.png       (Comprehensive 6-panel summary visualization)
"""

import os
import re
import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def parse_tasklog(log_path: str):
    """Parses task log for Phase 3 Gate Checks and Console Step prints."""
    gates_data = []
    console_steps = []

    if not os.path.exists(log_path):
        print(f"[Warn] Task log not found: {log_path}")
        return pd.DataFrame(), pd.DataFrame()

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Pattern for Phase 3 Gate Check
            # e.g.: >>> [Phase 3 Gate Check @ Step 500] Short Loss: 1.5569 | Long Loss: 1.7281 | Degradation Ratio: 1.110 | Gate Passed: NO
            gate_match = re.search(
                r">>>\s*\[Phase 3 Gate Check @ Step\s+(\d+)\]\s*Short Loss:\s*([\d\.]+)\s*\|\s*Long Loss:\s*([\d\.]+)\s*\|\s*Degradation Ratio:\s*([\d\.]+)\s*\|\s*Gate Passed:\s*(YES|NO)",
                line,
                re.IGNORECASE,
            )
            if gate_match:
                gates_data.append({
                    "step": int(gate_match.group(1)),
                    "short_mask_loss": float(gate_match.group(2)),
                    "long_mask_loss": float(gate_match.group(3)),
                    "degradation_ratio": float(gate_match.group(4)),
                    "gate_passed": gate_match.group(5).upper() == "YES",
                })

            # Pattern for Console Step prints
            # e.g.: Step  8400/10000 | Masked L1: 0.6772 | SNR: 16.87 dB | Eff. Rank: 122.2 (47.7%) | GradNorm: 0.49 | Throughput: 1340.2 aud-s/s | LR: 2.28e-05
            step_match = re.search(
                r"Step\s+(\d+)/(\d+)\s*\|\s*Masked L1:\s*([\d\.]+)\s*\|\s*SNR:\s*([\d\.\-]+)\s*dB\s*\|\s*Eff\.\s*Rank:\s*([\d\.]+)\s*\(([\d\.]+)%\)\s*\|\s*GradNorm:\s*([\d\.]+)\s*\|\s*Throughput:\s*([\d\.]+)\s*aud-s/s\s*\|\s*LR:\s*([\d\.eE\-+]+)",
                line,
            )
            if step_match:
                console_steps.append({
                    "step": int(step_match.group(1)),
                    "total_steps": int(step_match.group(2)),
                    "loss_masked_l1": float(step_match.group(3)),
                    "reconstruction_snr_db": float(step_match.group(4)),
                    "effective_rank": float(step_match.group(5)),
                    "rank_utilization_pct": float(step_match.group(6)),
                    "gradient_norm": float(step_match.group(7)),
                    "throughput_aud_sec_per_sec": float(step_match.group(8)),
                    "learning_rate": float(step_match.group(9)),
                })

    df_gates = pd.DataFrame(gates_data)
    df_console = pd.DataFrame(console_steps)
    return df_gates, df_console


def parse_jsonl(jsonl_path: str) -> pd.DataFrame:
    """Parses raw high-resolution metrics.jsonl into a Pandas DataFrame."""
    if not os.path.exists(jsonl_path):
        print(f"[Warn] JSONL file not found: {jsonl_path}")
        return pd.DataFrame()

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(records)
    if not df.empty and "step" in df.columns:
        df = df.sort_values(by="step").reset_index(drop=True)
    return df


def generate_publication_plots(df_full: pd.DataFrame, df_gates: pd.DataFrame, output_path: str):
    """Generates a multi-panel publication plot from the dumped metrics."""
    if df_full.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel 1: Loss curves
    ax = axes[0, 0]
    ax.plot(df_full["step"], df_full["loss_masked_l1"], label="Masked L1 Loss", color="#1f77b4", alpha=0.3)
    if len(df_full) > 50:
        smoothed_l1 = df_full["loss_masked_l1"].rolling(50, min_periods=1).mean()
        ax.plot(df_full["step"], smoothed_l1, label="Smoothed L1 (MA-50)", color="#1f77b4", linewidth=2.0)
    ax.set_title("Masked Spectrogram Reconstruction Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Panel 2: Reconstruction SNR
    ax = axes[0, 1]
    ax.plot(df_full["step"], df_full["reconstruction_snr_db"], color="#2ca02c", alpha=0.3, label="SNR (dB)")
    if len(df_full) > 50:
        smoothed_snr = df_full["reconstruction_snr_db"].rolling(50, min_periods=1).mean()
        ax.plot(df_full["step"], smoothed_snr, color="#2ca02c", linewidth=2.0, label="Smoothed SNR (MA-50)")
    ax.set_title("Reconstruction Signal-to-Noise Ratio (SNR)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reconstruction SNR (dB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    # Panel 3: Effective Hidden Rank
    ax = axes[1, 0]
    ax.plot(df_full["step"], df_full["effective_rank"], color="#9467bd", linewidth=1.5, label="Effective Rank (d_model=256)")
    ax.axhline(50.0, color="red", linestyle="--", linewidth=1.5, label="Intrinsic Gate Threshold (>50)")
    ax.set_title("Conformer Hidden Representation Rank", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Effective Rank")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    # Panel 4: Phase 3 Gate Degradation Ratio
    ax = axes[1, 1]
    if not df_gates.empty:
        passed_mask = df_gates["gate_passed"]
        ax.plot(df_gates["step"], df_gates["degradation_ratio"], marker="o", color="#ff7f0e", linewidth=2.0, label="Degradation Ratio (Long/Short)")
        ax.axhline(1.15, color="red", linestyle="--", linewidth=1.5, label="Gate Threshold (>1.15)")
        ax.scatter(df_gates[passed_mask]["step"], df_gates[passed_mask]["degradation_ratio"], color="green", s=80, label="Passed Gate", zorder=5)
        if (~passed_mask).any():
            ax.scatter(df_gates[~passed_mask]["step"], df_gates[~passed_mask]["degradation_ratio"], color="red", s=80, marker="x", label="Failed Gate", zorder=5)
    ax.set_title("Phase 3 Intrinsic Gate: Context Reliance Degradation", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss Degradation Ratio")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    # Panel 5: Throughput
    ax = axes[2, 0]
    if "throughput_audio_sec_per_sec" in df_full.columns:
        ax.plot(df_full["step"], df_full["throughput_audio_sec_per_sec"], color="#17becf", alpha=0.3, label="Throughput (Aud-s/s)")
        if len(df_full) > 50:
            smoothed_tp = df_full["throughput_audio_sec_per_sec"].rolling(50, min_periods=1).mean()
            ax.plot(df_full["step"], smoothed_tp, color="#17becf", linewidth=2.0, label="Smoothed Throughput")
    ax.set_title("Training Throughput (Audio-Seconds / Real-Second)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Audio Sec / Sec")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Panel 6: Cosine Similarity & Spectral Convergence
    ax = axes[2, 1]
    if "masked_cosine_similarity" in df_full.columns:
        ax.plot(df_full["step"], df_full["masked_cosine_similarity"], color="#e377c2", label="Masked Cosine Similarity", linewidth=1.5)
    if "spectral_convergence" in df_full.columns:
        ax.plot(df_full["step"], df_full["spectral_convergence"], color="#bcbd22", label="Spectral Convergence", linewidth=1.5)
    ax.set_title("Representation & Spectral Convergence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right")

    plt.suptitle("Marathi Conformer Pretraining (SSL) — Performance & Quality Telemetry", fontsize=15, fontweight="bold", y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Exported publication summary plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract training metrics to CSVs for publication plotting")
    parser.add_argument("--run_dir", type=str, default="runs/marathi_conformer_production", help="Run directory")
    parser.add_argument("--task_log", type=str, default=r"C:\Users\Machine Learning GPU\.gemini\antigravity-ide\brain\8f5aae47-4819-46bb-8a37-9e778a40c618\.system_generated\tasks\task-343.log", help="Path to task log file")
    parser.add_argument("--output_dir", type=str, default="runs/marathi_conformer_production/csv_metrics", help="Directory to save CSV files")
    parser.add_argument("--semantic_dir", type=str, default="checkpoints/pretrain", help="Directory to mirror checkpoint metrics")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir = Path(args.semantic_dir)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Extractor] Parsing task log: {args.task_log}")
    df_gates, df_console = parse_tasklog(args.task_log)

    jsonl_path = os.path.join(args.run_dir, "metrics.jsonl")
    print(f"[Extractor] Parsing metrics JSONL: {jsonl_path}")
    df_full = parse_jsonl(jsonl_path)

    # 1. Save Phase 3 Gates
    if not df_gates.empty:
        p3_path = out_dir / "pretrain_phase3_gates.csv"
        df_gates.to_csv(p3_path, index=False)
        df_gates.to_csv(semantic_dir / "pretrain_phase3_gates.csv", index=False)
        print(f"  [Dumped] {len(df_gates)} Phase 3 Gate evaluations -> {p3_path}")

    # 2. Save Console Steps
    if not df_console.empty:
        console_path = out_dir / "pretrain_tasklog_console_steps.csv"
        df_console.to_csv(console_path, index=False)
        df_console.to_csv(semantic_dir / "pretrain_tasklog_console_steps.csv", index=False)
        print(f"  [Dumped] {len(df_console)} Task log console steps -> {console_path}")

    # 3. Save Full Step Metrics
    if not df_full.empty:
        full_path = out_dir / "pretrain_full_step_metrics.csv"
        df_full.to_csv(full_path, index=False)
        df_full.to_csv(semantic_dir / "pretrain_full_step_metrics.csv", index=False)
        print(f"  [Dumped] {len(df_full)} Complete training step metrics -> {full_path}")

        # 4. Save Smoothed Metrics (Rolling 50 steps for clean plotting)
        numeric_cols = df_full.select_dtypes(include=["number"]).columns
        df_smoothed = df_full[numeric_cols].rolling(50, min_periods=1).mean()
        df_smoothed["step"] = df_full["step"]
        smoothed_path = out_dir / "pretrain_smoothed_50step_metrics.csv"
        df_smoothed.to_csv(smoothed_path, index=False)
        df_smoothed.to_csv(semantic_dir / "pretrain_smoothed_50step_metrics.csv", index=False)
        print(f"  [Dumped] Smoothed rolling metrics (50-step window) -> {smoothed_path}")

        # 5. Generate high-res publication plot
        plot_path = out_dir / "pretrain_publication_plots.png"
        generate_publication_plots(df_full, df_gates, str(plot_path))
        generate_publication_plots(df_full, df_gates, str(semantic_dir / "pretrain_publication_plots.png"))

    print("\n[Extractor] All metrics successfully dumped to CSV and plot generated!")


if __name__ == "__main__":
    main()
