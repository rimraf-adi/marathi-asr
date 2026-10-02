"""
Stage 5B: Direct Sequence-Level CER/WER Minimization via Group Relative Policy Optimization (GRPO).

Implements GRPO for Marathi Conformer ASR:
  1. Samples G candidate CTC paths per utterance (including greedy baseline + temperature stochastic rollouts).
  2. Collapses paths to text and evaluates non-differentiable sequence rewards:
       R(Y_i, Y*) = - (CER(Y_i, Y*) + gamma_wer * WER(Y_i, Y*))
  3. Computes normalized relative advantages within each group:
       A_i = (R_i - mean(R)) / (std(R) + eps)
  4. Updates policy parameters via PPO-style clipped surrogate objective:
       L_surr = - min(r_t * A, clip(r_t, 1-eps, 1+eps) * A)
  5. Enforces KL divergence constraint against frozen reference model:
       D_KL(pi_theta || pi_ref)
  6. Preserves acoustic-phonetic stability with an auxiliary supervised CTC anchor.
  7. Exhaustive telemetry logging: reward fluctuations, advantage distributions, CER/WER spreads,
     loss breakdowns, live 4-panel dashboard plotting to PNG, CSV, and JSONL.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    import jiwer
except ImportError:
    jiwer = None

try:
    import editdistance
except ImportError:
    editdistance = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils.tokenizer import MarathiTokenizer
from data_utils.utils import compute_cer, get_lr_scheduler
from moe.upcycling import load_moe_model
from moe.respin_dataset import build_or_load_speaker_split, LocalRESPINPrefetchLoader
from data_utils.kathbath_loader import load_kathbath_manifest, LocalKathbathPrefetchLoader
from telemetry.metrics_logger import RunMetricsLogger


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER) with fallback to word Levenshtein."""
    ref_clean = ref.strip()
    hyp_clean = hyp.strip()
    if len(ref_clean) == 0:
        return 1.0 if len(hyp_clean) > 0 else 0.0

    if jiwer is not None:
        try:
            return float(jiwer.wer(ref_clean, hyp_clean))
        except Exception:
            pass

    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    if editdistance is not None:
        return float(editdistance.eval(ref_words, hyp_words)) / len(ref_words)

    # Simple dynamic programming Levenshtein for words
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return float(dp[n][m]) / len(ref_words)


def collapse_ctc_path(path: List[int], blank_id: int = 0) -> List[int]:
    """Collapses consecutive identical tokens and strips blank tokens."""
    collapsed = []
    prev = -1
    for token in path:
        if token != prev:
            if token != blank_id:
                collapsed.append(token)
            prev = token
    return collapsed


def generate_live_plots(csv_path: Path, output_png: Path, stage_name: str):
    """
    Renders a comprehensive 4-panel publication-ready dashboard plotting:
      1. Reward Fluctuations & Rollout Spread (Mean, Min, Max, +/- Std Band)
      2. CER & WER Dynamics (Greedy vs Best-Hyp vs Worst-Hyp vs WER)
      3. Loss Breakdown (Total, GRPO Surrogate, KL Penalty, CTC Anchor)
      4. Policy Stability (Entropy, Ratio Clip Fraction, Gradient Norm)
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    steps, r_mean, r_std, r_min, r_max = [], [], [], [], []
    g_cer, b_cer, w_cer, m_wer = [], [], [], []
    l_tot, l_surr, l_kl, l_ctc = [], [], [], []
    p_ent, clip_frac, grad_norm, lrs = [], [], [], []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                r_mean.append(float(row.get("reward_mean", 0.0)))
                r_std.append(float(row.get("reward_std", 0.0)))
                r_min.append(float(row.get("reward_min", 0.0)))
                r_max.append(float(row.get("reward_max", 0.0)))

                g_cer.append(float(row.get("greedy_cer", 0.0)) * 100.0)
                b_cer.append(float(row.get("best_hyp_cer", 0.0)) * 100.0)
                w_cer.append(float(row.get("worst_hyp_cer", 0.0)) * 100.0)
                m_wer.append(float(row.get("mean_hyp_wer", 0.0)) * 100.0)

                l_tot.append(float(row.get("loss_total", 0.0)))
                l_surr.append(float(row.get("loss_grpo_surr", 0.0)))
                l_kl.append(float(row.get("loss_kl", 0.0)))
                l_ctc.append(float(row.get("loss_ctc", 0.0)))

                p_ent.append(float(row.get("policy_entropy", 0.0)))
                clip_frac.append(float(row.get("clip_fraction", 0.0)) * 100.0)
                grad_norm.append(float(row.get("grad_norm", 0.0)))
                lrs.append(float(row.get("lr", 0.0)))
    except Exception as e:
        print(f"[Plotting Warning] Could not parse CSV for plotting: {e}")
        return

    if len(steps) < 2:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"GRPO Sequence Polish Telemetry Dashboard ({stage_name})", fontsize=15, fontweight="bold")

    # Panel 1: Reward Fluctuations
    axs[0, 0].set_title("1. Reward Fluctuations & Rollout Spread", fontweight="bold", fontsize=11)
    r_m = np.array(r_mean)
    r_s = np.array(r_std)
    axs[0, 0].plot(steps, r_m, color="#1f77b4", linewidth=2.0, label="Mean Reward (Group)")
    axs[0, 0].fill_between(steps, r_m - r_s, r_m + r_s, color="#1f77b4", alpha=0.22, label=r"$\pm 1\sigma$ Spread")
    axs[0, 0].plot(steps, r_max, color="#2ca02c", linestyle="--", alpha=0.7, label="Max Reward (Best Rollout)")
    axs[0, 0].plot(steps, r_min, color="#d62728", linestyle=":", alpha=0.7, label="Min Reward (Worst Rollout)")
    axs[0, 0].set_xlabel("Steps")
    axs[0, 0].set_ylabel("Reward Value: -(CER + gamma*WER)")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend(loc="lower right", fontsize=8)

    # Panel 2: CER & WER Dynamics
    axs[0, 1].set_title("2. Error Rate Reduction (CER & WER %)", fontweight="bold", fontsize=11)
    axs[0, 1].plot(steps, g_cer, color="#ff7f0e", linewidth=2.0, label="Greedy CER (Decoded)")
    axs[0, 1].plot(steps, b_cer, color="#2ca02c", linewidth=1.5, linestyle="-.", label="Best Rollout CER")
    axs[0, 1].plot(steps, w_cer, color="#d62728", linewidth=1.2, linestyle=":", label="Worst Rollout CER")
    axs[0, 1].plot(steps, m_wer, color="#9467bd", linewidth=1.5, label="Group Mean WER %")
    axs[0, 1].set_xlabel("Steps")
    axs[0, 1].set_ylabel("Error Rate (%)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend(loc="upper right", fontsize=8)

    # Panel 3: Loss Breakdown
    axs[1, 0].set_title("3. Multi-Component Loss Breakdown", fontweight="bold", fontsize=11)
    axs[1, 0].plot(steps, l_tot, color="#333333", linewidth=2.0, label="Total Loss")
    axs[1, 0].plot(steps, l_surr, color="#e377c2", linewidth=1.5, label="GRPO Surrogate Loss")
    axs[1, 0].plot(steps, l_kl, color="#17becf", linewidth=1.5, label="Reference KL Divergence")
    axs[1, 0].plot(steps, l_ctc, color="#bcbd22", linewidth=1.2, linestyle="--", label="Auxiliary CTC Anchor")
    axs[1, 0].set_xlabel("Steps")
    axs[1, 0].set_ylabel("Loss Magnitude")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend(loc="upper right", fontsize=8)

    # Panel 4: Policy Stability Diagnostics
    axs[1, 1].set_title("4. Policy Diagnostics & Clipping Rate", fontweight="bold", fontsize=11)
    ax4_sub = axs[1, 1].twinx()
    p1 = axs[1, 1].plot(steps, p_ent, color="#8c564b", linewidth=1.8, label="Policy Entropy")
    p2 = axs[1, 1].plot(steps, grad_norm, color="#7f7f7f", linestyle=":", label="Gradient Norm")
    p3 = ax4_sub.plot(steps, clip_frac, color="#d62728", linewidth=1.5, linestyle="--", label="Ratio Clip Frac (%)")
    axs[1, 1].set_xlabel("Steps")
    axs[1, 1].set_ylabel("Entropy / Grad Norm")
    ax4_sub.set_ylabel("Clipped Tokens (%)", color="#d62728")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    axs[1, 1].legend(lines, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close(fig)


def train_grpo_stage5(
    run_dir: str = "runs/run_grpo_polish",
    moe_ckpt: str = "runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt",
    total_steps: int = 2000,
    batch_size: int = 4,
    group_size: int = 4,
    temperature: float = 0.85,
    gamma_wer: float = 0.20,
    clip_eps: float = 0.20,
    beta_kl: float = 0.04,
    alpha_ctc: float = 0.15,
    lr: float = 2e-5,
    min_lr: float = 2e-6,
    warmup_steps: int = 150,
    grad_accum_steps: int = 2,
    log_every: int = 10,
    plot_every: int = 25,
    save_every: int = 500,
    checkpoint_dir: Optional[str] = None,
    stage_name: str = "stage5_grpo_polish",
    resume_path: Optional[str] = None,
    train_top_layers: int = 8,  # layers 5-12 (index 4..11) + CTC head
):
    """
    Main GRPO Training Routine for Stage 5 Sequence Polishing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"[{stage_name}] GRPO Sequence Polish Engine on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Configuration:")
    print(f"    - Base Checkpoint   : {moe_ckpt}")
    print(f"    - Steps             : {total_steps}")
    print(f"    - Batch Size        : {batch_size} (Grad Accum: {grad_accum_steps}, Effective: {batch_size * grad_accum_steps})")
    print(f"    - Group Size (G)    : {group_size} rollouts per utterance")
    print(f"    - Sampling Temp     : {temperature:.2f}")
    print(f"    - Reward Weights    : R = - (CER + {gamma_wer:.2f} * WER)")
    print(f"    - Policy Clipping   : eps = {clip_eps:.2f}")
    print(f"    - KL Penalty (beta) : {beta_kl:.4f}")
    print(f"    - CTC Anchor (alpha): {alpha_ctc:.4f}")
    print(f"    - Learning Rate     : {lr:.2e} -> {min_lr:.2e} (Warmup: {warmup_steps})")
    print("=" * 80 + "\n")

    actual_run_dir = Path(run_dir)
    actual_ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else actual_run_dir / "checkpoints" / stage_name
    plots_dir = actual_run_dir / "plots"
    actual_ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    print(f"[Tokenizer] Loaded vocabulary with {tokenizer.vocab_size} tokens")

    # 1. Load Trainable Policy Model (pi_theta)
    print(f"[Policy Model] Initializing policy network from: {moe_ckpt}")
    model = load_moe_model(moe_ckpt, device=device)

    # 2. Load Frozen Reference Model (pi_ref) for KL Penalty
    print(f"[Reference Model] Initializing frozen reference network from: {moe_ckpt}")
    ref_model = load_moe_model(moe_ckpt, device=device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3. Parameter Freezing Strategy for Policy Model
    print(f"[Freezing Strategy] Freezing bottom {12 - train_top_layers} Conformer layers (Acoustic grounding anchor)...")
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []
    # Unfreeze top Conformer blocks (e.g. layers 5-12, indexing 4..11)
    start_layer = 12 - train_top_layers
    for layer_idx in range(start_layer, 12):
        for param in model.encoder.layers[layer_idx].parameters():
            param.requires_grad = True
            trainable_params.append(param)

    # Unfreeze CTC projection head
    for param in model.ctc_head.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"[Parameters] Optimizing {num_trainable:,} / {num_total:,} parameters ({100 * num_trainable / num_total:.2f}%)")

    # 4. Load OpenSLR64 Dataset (For Final Alignment / Sequence Polish)
    print("[Data] Ingesting OpenSLR64 manifest...")
    slr_manifest = "data/openslr64_marathi/manifest.json"
    slr_dir = "data/openslr64_marathi"
    calib_utts = load_kathbath_manifest(slr_manifest, slr_dir)
    total_calib_hours = sum(u["duration"] for u in calib_utts) / 3600.0
    print(f"[Data] OpenSLR64 dataset verified: {len(calib_utts):,} utterances ({total_calib_hours:.1f} hours)")

    loader = LocalKathbathPrefetchLoader(calib_utts, tokenizer, batch_size=batch_size, queue_size=4)

    # 5. Optimization & Telemetry Setup
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr / lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    ctc_criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    logger = RunMetricsLogger(
        stage_name=stage_name,
        run_dir=run_dir,
        total_steps=total_steps,
        plot_interval=plot_every,
    )
    dashboard_png = plots_dir / f"{stage_name}_dashboard.png"

    step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint state from: {resume_path}")
        r_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(r_ckpt["model_state_dict"])
        optimizer.load_state_dict(r_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(r_ckpt["scheduler_state_dict"])
        step = r_ckpt.get("step", 0)

    model.train()
    start_time = time.time()
    rolling_metrics = {}

    print(f"\n[{stage_name}] Commencing GRPO Training Run (Target Steps: {total_steps})...\n")

    try:
        while step < total_steps:
            step_start = time.time()
            batch = next(loader)
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = (batch.get("audio_lens") if "audio_lens" in batch else batch["audio_lengths"]).to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lens = (batch.get("target_lens") if "target_lens" in batch else batch["target_lengths"]).to(device, non_blocking=True)
            ref_texts = batch["texts"]

            b_curr = audio.size(0)
            step += 1

            # -----------------------------------------------------------------
            # Step A: Forward Pass through Reference Model (Frozen)
            # -----------------------------------------------------------------
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    ref_ctc_dict = ref_model.forward_ctc(audio, chunk_size=None)
                    ref_log_probs = ref_ctc_dict["log_probs"]  # (B, T, V)

            # -----------------------------------------------------------------
            # Step B: Forward Pass through Current Policy Model
            # -----------------------------------------------------------------
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # Pass through backbone
                spec = model.frontend(audio)
                enc_out = model.encoder(spec, return_all_exits=False)
                final_hidden = enc_out["final_hidden"]  # (B, T, d_model)

                # Extract unnormalized logits from CTC Head
                normed_hidden = model.ctc_head.norm(final_hidden)
                dropout_hidden = model.ctc_head.dropout(normed_hidden)
                logits = model.ctc_head.proj(dropout_hidden)  # (B, T, V)

                log_probs = F.log_softmax(logits, dim=-1)     # (B, T, V)
                probs = F.softmax(logits, dim=-1)             # (B, T, V)

            t_frames = logits.size(1)
            valid_lens = torch.clamp((audio_lens // 160 // 4), max=t_frames)

            # -----------------------------------------------------------------
            # Step C: Group Rollout Sampling (G candidates per utterance)
            #   Candidate 0: Greedy argmax decode (deterministic anchor)
            #   Candidates 1..G-1: Stochastic sampling with temperature
            # -----------------------------------------------------------------
            with torch.no_grad():
                rollout_paths = []    # list of length B, each containing G lists of token IDs
                rollout_texts = []    # list of length B, each containing G string hypotheses
                rollout_rewards = []  # list of length B, each containing G float rewards
                rollout_cers = []     # list of length B, each containing G float CERs
                rollout_wers = []     # list of length B, each containing G float WERs

                # Temperature scaled float32 logits for numerically stable sampling
                scaled_logits = logits.float() / max(temperature, 1e-4)

                for b_i in range(b_curr):
                    t_len = int(valid_lens[b_i].item())
                    b_logits = logits[b_i, :t_len]
                    b_scaled_logits = scaled_logits[b_i, :t_len]
                    ref_t = ref_texts[b_i]

                    paths_b = []
                    texts_b = []
                    rewards_b = []
                    cers_b = []
                    wers_b = []

                    # Build categorical distribution once per utterance in float32
                    dist = torch.distributions.Categorical(logits=b_scaled_logits)

                    for g_i in range(group_size):
                        if g_i == 0:
                            # Candidate 0: Greedy Argmax (Exploitation anchor)
                            path_tokens = torch.argmax(b_logits, dim=-1).tolist()
                        else:
                            # Candidates 1..G-1: Categorical Temperature Sampling (Exploration)
                            path_tokens = dist.sample().tolist()

                        paths_b.append(path_tokens)

                        # CTC collapse & decode
                        collapsed = collapse_ctc_path(path_tokens, blank_id=tokenizer.blank_id)
                        hyp_text = tokenizer.ctc_decode(collapsed)
                        texts_b.append(hyp_text)

                        # Sequence Reward Computation
                        cer_val = compute_cer(ref_t, hyp_text)
                        wer_val = compute_wer(ref_t, hyp_text)
                        reward_val = - (cer_val + gamma_wer * wer_val)

                        cers_b.append(cer_val)
                        wers_b.append(wer_val)
                        rewards_b.append(reward_val)

                    rollout_paths.append(paths_b)
                    rollout_texts.append(texts_b)
                    rollout_rewards.append(rewards_b)
                    rollout_cers.append(cers_b)
                    rollout_wers.append(wers_b)

            # -----------------------------------------------------------------
            # Step D: Normalized Group Advantages (GRPO)
            # -----------------------------------------------------------------
            # rewards_tensor: (B, G)
            rewards_tensor = torch.tensor(rollout_rewards, dtype=torch.float32, device=device)
            group_mean = rewards_tensor.mean(dim=-1, keepdim=True)
            group_std = rewards_tensor.std(dim=-1, keepdim=True) + 1e-8
            advantages = (rewards_tensor - group_mean) / group_std  # (B, G)

            # -----------------------------------------------------------------
            # Step E: GRPO Clipped Surrogate Loss over Sampled Trajectories
            # -----------------------------------------------------------------
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # Prepare tensor indices for gathering log-probs
                total_surrogate_loss = torch.tensor(0.0, device=device)
                num_rollouts = 0
                clip_count = 0
                total_tokens = 0

                # Compute baseline log-probs at rollout time (behavior policy)
                old_log_probs = log_probs.detach()

                for b_i in range(b_curr):
                    t_len = int(valid_lens[b_i].item())
                    if t_len == 0:
                        continue

                    # Slice time frames for current sample
                    b_log_probs = log_probs[b_i, :t_len]          # (T, V)
                    b_old_log_probs = old_log_probs[b_i, :t_len]  # (T, V)

                    for g_i in range(group_size):
                        path_tensor = torch.tensor(rollout_paths[b_i][g_i], dtype=torch.long, device=device)  # (T,)
                        adv = advantages[b_i, g_i]

                        # Current and old action log-probabilities along the trajectory (in float32)
                        curr_token_log_p = b_log_probs.gather(dim=-1, index=path_tensor.unsqueeze(-1)).squeeze(-1).float()  # (T,)
                        old_token_log_p = b_old_log_probs.gather(dim=-1, index=path_tensor.unsqueeze(-1)).squeeze(-1).float()  # (T,)

                        # Policy Ratio: r_t(theta) = exp(log_p - log_p_old)
                        ratio = torch.exp(curr_token_log_p - old_token_log_p)

                        # Clipped PPO Objective
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                        trajectory_surrogate = - torch.min(surr1, surr2).mean()

                        total_surrogate_loss = total_surrogate_loss + trajectory_surrogate
                        num_rollouts += 1

                        # Track clipping fraction
                        with torch.no_grad():
                            clipped = (torch.abs(ratio - 1.0) > clip_eps).sum().item()
                            clip_count += clipped
                            total_tokens += t_len

                loss_surrogate = total_surrogate_loss / max(num_rollouts, 1)

                # -----------------------------------------------------------------
                # Step F: KL Divergence Penalty against Frozen Reference Model
                # -----------------------------------------------------------------
                # Categorical KL over vocabulary: sum(p_ref * (log_p_ref - log_p_curr))
                kl_per_frame = F.kl_div(log_probs, ref_log_probs, reduction="none", log_target=True).sum(dim=-1)  # (B, T)

                # Mask out padded frames
                mask = torch.arange(t_frames, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)  # (B, T)
                loss_kl = (kl_per_frame * mask).sum() / max(mask.sum(), 1.0)

                # -----------------------------------------------------------------
                # Step G: Auxiliary Supervised CTC Anchor Loss
                # -----------------------------------------------------------------
                log_probs_t = log_probs.transpose(0, 1)  # (T, B, V)
                loss_ctc = ctc_criterion(log_probs_t, targets, valid_lens, target_lens)

                # -----------------------------------------------------------------
                # Total Combined Objective
                # -----------------------------------------------------------------
                total_loss = loss_surrogate + (beta_kl * loss_kl) + (alpha_ctc * loss_ctc)

            # Backward pass with Gradient Scaling
            loss_scaled = total_loss / grad_accum_steps
            scaler.scale(loss_scaled).backward()

            grad_norm_val = 0.0
            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                grad_norm_val = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # -----------------------------------------------------------------
            # Step H: Collect Comprehensive Telemetry Metrics
            # -----------------------------------------------------------------
            step_time = time.time() - step_start
            batch_audio_sec = float(audio_lens.sum().item()) / 16000.0
            throughput = batch_audio_sec / max(step_time, 1e-5)

            with torch.no_grad():
                # Policy entropy: - sum(p * log_p)
                frame_entropy = - (probs * log_probs).sum(dim=-1)
                policy_entropy = float(((frame_entropy * mask).sum() / max(mask.sum(), 1.0)).item())

                # Flatten rollout statistics for batch metrics
                all_rewards = [r for sub in rollout_rewards for r in sub]
                all_cers = [c for sub in rollout_cers for c in sub]
                all_wers = [w for sub in rollout_wers for w in sub]
                greedy_cers = [sub[0] for sub in rollout_cers]
                best_cers = [min(sub) for sub in rollout_cers]
                worst_cers = [max(sub) for sub in rollout_cers]

                step_metrics = {
                    # Rewards
                    "reward_mean": round(float(np.mean(all_rewards)), 4),
                    "reward_std": round(float(np.mean([np.std(sub) for sub in rollout_rewards])), 4),
                    "reward_min": round(float(np.min(all_rewards)), 4),
                    "reward_max": round(float(np.max(all_rewards)), 4),
                    # Error Rates
                    "greedy_cer": round(float(np.mean(greedy_cers)), 4),
                    "best_hyp_cer": round(float(np.mean(best_cers)), 4),
                    "worst_hyp_cer": round(float(np.mean(worst_cers)), 4),
                    "mean_hyp_cer": round(float(np.mean(all_cers)), 4),
                    "mean_hyp_wer": round(float(np.mean(all_wers)), 4),
                    "cer_spread": round(float(np.mean(worst_cers) - np.mean(best_cers)), 4),
                    # Losses
                    "loss_total": round(float(total_loss.item()), 4),
                    "loss_grpo_surr": round(float(loss_surrogate.item()), 4),
                    "loss_kl": round(float(loss_kl.item()), 4),
                    "loss_ctc": round(float(loss_ctc.item()), 4),
                    # Policy Diagnostics
                    "clip_fraction": round(float(clip_count / max(total_tokens, 1)), 4),
                    "policy_entropy": round(policy_entropy, 3),
                    "advantage_mean": round(float(advantages.mean().item()), 4),
                    "advantage_std": round(float(advantages.std().item()), 4),
                    "grad_norm": round(grad_norm_val, 3),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "throughput_aud_s_per_s": round(throughput, 1),
                    "step_time_sec": round(step_time, 2),
                }

            # Log to CSV and JSONL
            logger.log_step(step, step_metrics)

            # Periodic Terminal Reporting
            if step % log_every == 0 or step == 1:
                cur_lr = step_metrics["lr"]
                print(
                    f"Step {step:5d}/{total_steps} | GRPO Polish | "
                    f"R_mean: {step_metrics['reward_mean']:+.3f} (std: {step_metrics['reward_std']:.3f}, [{step_metrics['reward_min']:+.2f}, {step_metrics['reward_max']:+.2f}]) | "
                    f"Greedy CER: {step_metrics['greedy_cer']*100:5.2f}% (Best: {step_metrics['best_hyp_cer']*100:5.2f}%) | "
                    f"Surr: {step_metrics['loss_grpo_surr']:+.3f} | KL: {step_metrics['loss_kl']:.4f} | CTC: {step_metrics['loss_ctc']:.3f} | "
                    f"Clip: {step_metrics['clip_fraction']*100:4.1f}% | Ent: {step_metrics['policy_entropy']:.2f} | "
                    f"Throughput: {throughput:5.1f} a-s/s"
                )
                print(f"  [Ref  ]: {ref_texts[0][:75]}")
                print(f"  [Greedy]: {rollout_texts[0][0][:75]}")
                print(f"  [Best  ]: {rollout_texts[0][np.argmax(rollout_rewards[0])][:75]}\n")

            # Periodic Live Plot Generation
            if step % plot_every == 0 or step == 1:
                generate_live_plots(logger.csv_path, dashboard_png, stage_name)

            # Periodic Checkpoint Saving
            if step % save_every == 0 or step == total_steps:
                logger.save_checkpoint(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_val=step_metrics["greedy_cer"],
                    metric_name="greedy_cer",
                    lower_is_better=True,
                    metadata={
                        "reward_mean": step_metrics["reward_mean"],
                        "best_hyp_cer": step_metrics["best_hyp_cer"],
                        "loss_total": step_metrics["loss_total"],
                    },
                )
                print(f"  [Checkpoint] Step {step} saved to: {logger.ckpt_dir}")

    finally:
        loader.close()

    total_time = (time.time() - start_time) / 3600.0
    generate_live_plots(logger.csv_path, dashboard_png, stage_name)
    print(f"\n[Completed] GRPO Sequence Polish finished in {total_time:.2f} hours!")
    print(f"  Final Dashboard Plot: {dashboard_png.resolve()}")
    print(f"  Final Metrics CSV   : {logger.csv_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5B: Direct Sequence Polish via GRPO")
    parser.add_argument("--moe_ckpt", type=str, default="runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt", help="MoE checkpoint to polish")
    parser.add_argument("--steps", type=int, default=2000, help="Total GRPO steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--group_size", type=int, default=4, help="Group size (G rollouts per utterance)")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    parser.add_argument("--gamma_wer", type=float, default=0.20, help="WER penalty coefficient in reward")
    parser.add_argument("--clip_eps", type=float, default=0.20, help="PPO clipping epsilon")
    parser.add_argument("--beta_kl", type=float, default=0.04, help="KL penalty coefficient against reference model")
    parser.add_argument("--alpha_ctc", type=float, default=0.15, help="Auxiliary CTC loss coefficient")
    parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=2e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Warmup steps")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--log_every", type=int, default=10, help="Terminal log interval")
    parser.add_argument("--plot_every", type=int, default=25, help="Dashboard plot interval")
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--run_dir", type=str, default="runs/run_grpo_polish", help="Output run directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Custom checkpoint output directory")
    parser.add_argument("--stage_name", type=str, default="stage5_grpo_polish", help="Stage identifier")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--train_top_layers", type=int, default=8, help="Number of top Conformer layers to fine-tune")

    args = parser.parse_args()

    train_grpo_stage5(
        run_dir=args.run_dir,
        moe_ckpt=args.moe_ckpt,
        total_steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        temperature=args.temperature,
        gamma_wer=args.gamma_wer,
        clip_eps=args.clip_eps,
        beta_kl=args.beta_kl,
        alpha_ctc=args.alpha_ctc,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        plot_every=args.plot_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        stage_name=args.stage_name,
        resume_path=args.resume,
        train_top_layers=args.train_top_layers,
    )
