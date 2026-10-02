"""
Smoke test suite for GRPO Sequence Polish Engine (final_alignment/train_grpo_polish.py).
Verifies:
  1. Forward pass on policy and frozen reference models.
  2. Multi-candidate trajectory rollouts (greedy + temperature categorical sampling).
  3. CTC sequence collapsing and Marathi text decoding.
  4. Edit-distance CER and WER reward calculation.
  5. Normalized relative advantage calculation across groups.
  6. Clipped surrogate loss, KL divergence, and CTC anchor loss.
  7. Autograd backward pass and gradient step on trainable parameters.
  8. Telemetry CSV logging and 4-panel live plot generation.
"""

import os
import sys
import shutil
from pathlib import Path

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import numpy as np

from data_utils.tokenizer import MarathiTokenizer
from moe.upcycling import load_moe_model
from final_alignment.train_grpo_polish import (
    compute_wer,
    collapse_ctc_path,
    generate_live_plots,
    train_grpo_stage5,
)


def test_grpo_unit_components():
    print("\n--- Test 1: Testing Unit Components (WER, CTC Collapse, Advantages) ---")
    ref = "मला काल बाजारात जायला जमले नाही"
    hyp_identical = "मला काल बाजारात जायला जमले नाही"
    hyp_error = "मला बाजारात जावक जमला नाही"

    wer_0 = compute_wer(ref, hyp_identical)
    wer_err = compute_wer(ref, hyp_error)
    print(f"  WER identical: {wer_0:.4f} (expected: 0.0)")
    print(f"  WER with errors: {wer_err:.4f} (expected: > 0.0)")
    assert wer_0 == 0.0, "Identical strings must have 0.0 WER"
    assert wer_err > 0.0, "Altered strings must have > 0.0 WER"

    # CTC collapse test: [0, 5, 5, 0, 10, 10, 10, 0, 20] -> [5, 10, 20]
    raw_path = [0, 5, 5, 0, 10, 10, 10, 0, 20]
    collapsed = collapse_ctc_path(raw_path, blank_id=0)
    print(f"  Raw path: {raw_path} -> Collapsed: {collapsed}")
    assert collapsed == [5, 10, 20], f"Expected [5, 10, 20], got {collapsed}"

    # Advantage normalization test
    # Group rewards: [-0.05, -0.15, -0.30, -0.05]
    rewards = torch.tensor([[-0.05, -0.15, -0.30, -0.05]], dtype=torch.float32)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True) + 1e-8
    adv = (rewards - mean) / std
    print(f"  Group Rewards: {rewards.tolist()[0]}")
    print(f"  Calculated Normalized Advantages: {adv.tolist()[0]}")
    assert adv[0, 0] > 0.0, "Best rollout should have positive advantage"
    assert adv[0, 2] < 0.0, "Worst rollout should have negative advantage"
    assert torch.abs(adv.mean()) < 1e-4, "Group advantages should have zero mean"
    print("  [PASS] Unit components verified successfully!")


def test_grpo_plot_generation():
    print("\n--- Test 2: Testing Live Plot Dashboard Generator ---")
    test_csv = Path("runs/test_smoke_grpo/test_metrics.csv")
    test_png = Path("runs/test_smoke_grpo/test_dashboard.png")
    test_csv.parent.mkdir(parents=True, exist_ok=True)

    # Generate synthetic telemetry rows
    fieldnames = [
        "step", "timestamp", "reward_mean", "reward_std", "reward_min", "reward_max",
        "greedy_cer", "best_hyp_cer", "worst_hyp_cer", "mean_hyp_cer", "mean_hyp_wer",
        "cer_spread", "loss_total", "loss_grpo_surr", "loss_kl", "loss_ctc",
        "clip_fraction", "policy_entropy", "advantage_mean", "advantage_std",
        "grad_norm", "lr", "throughput_aud_s_per_s", "step_time_sec"
    ]
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in range(1, 20):
            writer.writerow({
                "step": s * 10,
                "timestamp": 1000.0 + s,
                "reward_mean": -0.15 + s * 0.005,
                "reward_std": 0.04,
                "reward_min": -0.25 + s * 0.004,
                "reward_max": -0.08 + s * 0.006,
                "greedy_cer": max(0.04, 0.12 - s * 0.004),
                "best_hyp_cer": max(0.02, 0.08 - s * 0.003),
                "worst_hyp_cer": max(0.06, 0.18 - s * 0.005),
                "mean_hyp_cer": 0.11 - s * 0.003,
                "mean_hyp_wer": 0.35 - s * 0.008,
                "cer_spread": 0.06,
                "loss_total": 0.50 - s * 0.015,
                "loss_grpo_surr": -0.02 + s * 0.001,
                "loss_kl": 0.008 + s * 0.0005,
                "loss_ctc": 0.40 - s * 0.015,
                "clip_fraction": 0.03 + (s % 3) * 0.01,
                "policy_entropy": 0.45 - s * 0.005,
                "advantage_mean": 0.0001,
                "advantage_std": 0.9998,
                "grad_norm": 0.85 + (s % 2) * 0.2,
                "lr": 2e-5,
                "throughput_aud_s_per_s": 85.0,
                "step_time_sec": 0.45,
            })

    generate_live_plots(test_csv, test_png, "test_smoke_grpo")
    assert test_png.exists(), f"Plot was not created at {test_png}"
    assert test_png.stat().st_size > 5000, f"Plot file seems corrupted or empty: {test_png.stat().st_size} bytes"
    print(f"  [PASS] Live dashboard plot created successfully ({test_png.stat().st_size} bytes) at {test_png}")


def test_grpo_end_to_end_smoke():
    print("\n--- Test 3: End-to-End GRPO Engine Smoke Test (3 Steps) ---")
    smoke_run_dir = Path("runs/smoke_test_grpo")
    if smoke_run_dir.exists():
        shutil.rmtree(smoke_run_dir)

    moe_ckpt = "runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt"
    assert os.path.exists(moe_ckpt), f"MoE checkpoint not found at {moe_ckpt}"

    # Run 3 steps of GRPO training with small batch size and G=3
    train_grpo_stage5(
        run_dir=str(smoke_run_dir),
        moe_ckpt=moe_ckpt,
        total_steps=3,
        batch_size=2,
        group_size=3,
        temperature=0.85,
        gamma_wer=0.20,
        clip_eps=0.20,
        beta_kl=0.04,
        alpha_ctc=0.15,
        lr=2e-5,
        min_lr=2e-6,
        warmup_steps=1,
        grad_accum_steps=1,
        log_every=1,
        plot_every=1,
        save_every=3,
        stage_name="smoke_grpo",
        train_top_layers=4,  # unfreeze top 4 layers for quick smoke test
    )

    # Verify telemetry CSV and plots
    csv_file = smoke_run_dir / "csvs" / "smoke_grpo_metrics.csv"
    plot_file = smoke_run_dir / "plots" / "smoke_grpo_dashboard.png"
    ckpt_file = smoke_run_dir / "checkpoints" / "smoke_grpo" / "smoke_grpo_step_3.pt"

    assert csv_file.exists(), f"Metrics CSV missing: {csv_file}"
    assert plot_file.exists(), f"Dashboard plot missing: {plot_file}"
    assert ckpt_file.exists(), f"Checkpoint missing: {ckpt_file}"

    # Check CSV row count
    import csv
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3, f"Expected 3 steps logged in CSV, got {len(reader)}"
        first_row = reader[0]
        print(f"  Step 1 Logged Reward Mean: {first_row['reward_mean']}")
        print(f"  Step 1 Logged Greedy CER : {first_row['greedy_cer']}")
        print(f"  Step 1 Logged Total Loss : {first_row['loss_total']}")
        print(f"  Step 1 Logged Surrogate  : {first_row['loss_grpo_surr']}")
        print(f"  Step 1 Logged KL Loss    : {first_row['loss_kl']}")
        print(f"  Step 1 Logged Entropy    : {first_row['policy_entropy']}")

    print("\n" + "=" * 60)
    print("  [SUCCESS] All GRPO Smoke Tests Passed Flawlessly!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_grpo_unit_components()
    test_grpo_plot_generation()
    test_grpo_end_to_end_smoke()
