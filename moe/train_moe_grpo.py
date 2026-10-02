"""
Stage 4B: MoE Dialect Expert Routing Optimization via Group Relative Policy Optimization (GRPO).

Replaces heuristic load-balancing with performance-driven RL routing:
  1. For each utterance, evaluates G=3 candidate expert assignments across regional dialects:
       Expert 0: D1 (Malvani / Konkan)
       Expert 1: D2 (Ahirani / Khandesh)
       Expert 2: D4 (Varhadi / Vidarbha)
  2. Measures sequence transcription reward:
       R(k_i) = - (CER(Y^{(k_i)}, Y*) + gamma_wer * WER(Y^{(k_i)}, Y*)) + eta * I(k_i == dialect_idx)
  3. Computes normalized relative advantages across the 3 dialect experts:
       A(k_i) = (R(k_i) - mean(R)) / (std(R) + eps)
  4. Updates the MoE routing policy via PPO-style clipped surrogate objective:
       L_surr = - min(r_k * A_k, clip(r_k, 1-eps, 1+eps) * A_k)
  5. Enforces router entropy regularization and supervised CTC anchor loss.
  6. Comprehensive telemetry: tracks reward fluctuations, per-dialect CER, expert load distribution,
     oracle routing gap, live 4-panel dashboard plotting to PNG, CSV, and JSONL.
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


DIALECT_NAMES = {
    0: "D1 (Malvani)",
    1: "D2 (Ahirani)",
    2: "D4 (Varhadi)",
    3: "D3 (Standard)",
}


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER) with fallback."""
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
    """Collapses consecutive duplicates and removes blank tokens."""
    collapsed = []
    prev = -1
    for token in path:
        if token != prev:
            if token != blank_id:
                collapsed.append(token)
            prev = token
    return collapsed


def generate_moe_grpo_plots(csv_path: Path, output_png: Path, stage_name: str):
    """
    Renders a comprehensive 4-panel publication-ready dashboard:
      1. Reward Dynamics & Rollout Spread (Mean, Min, Max, +/- Std Band)
      2. Per-Expert CER & Dialect Accuracy (E0, E1, E2 CER vs Accuracy %)
      3. Router Load Balancing Dynamics (E0 Malvani, E1 Ahirani, E2 Varhadi shares)
      4. Loss Decomposition & Router Entropy
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    steps, r_mean, r_std, r_min, r_max = [], [], [], [], []
    cer_e0, cer_e1, cer_e2, d_acc = [], [], [], []
    load_e0, load_e1, load_e2 = [], [], []
    l_tot, l_surr, l_ctc, r_ent = [], [], [], []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                r_mean.append(float(row.get("reward_mean", 0.0)))
                r_std.append(float(row.get("reward_std", 0.0)))
                r_min.append(float(row.get("reward_min", 0.0)))
                r_max.append(float(row.get("reward_max", 0.0)))

                cer_e0.append(float(row.get("expert_cer_e0", 0.0)) * 100.0)
                cer_e1.append(float(row.get("expert_cer_e1", 0.0)) * 100.0)
                cer_e2.append(float(row.get("expert_cer_e2", 0.0)) * 100.0)
                d_acc.append(float(row.get("dialect_acc", 0.0)) * 100.0)

                load_e0.append(float(row.get("router_load_e0", 0.33)) * 100.0)
                load_e1.append(float(row.get("router_load_e1", 0.33)) * 100.0)
                load_e2.append(float(row.get("router_load_e2", 0.33)) * 100.0)

                l_tot.append(float(row.get("loss_total", 0.0)))
                l_surr.append(float(row.get("loss_grpo_surr", 0.0)))
                l_ctc.append(float(row.get("loss_ctc", 0.0)))
                r_ent.append(float(row.get("router_entropy", 0.0)))
    except Exception as e:
        print(f"[Plotting Warning] Could not parse CSV for plotting: {e}")
        return

    if len(steps) < 2:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"MoE Dialect Router GRPO Telemetry Dashboard ({stage_name})", fontsize=15, fontweight="bold")

    # Panel 1: Reward Fluctuations
    axs[0, 0].set_title("1. Router Reward Fluctuations & Spread", fontweight="bold", fontsize=11)
    r_m = np.array(r_mean)
    r_s = np.array(r_std)
    axs[0, 0].plot(steps, r_m, color="#1f77b4", linewidth=2.0, label="Mean Reward (Group G=3)")
    axs[0, 0].fill_between(steps, r_m - r_s, r_m + r_s, color="#1f77b4", alpha=0.22, label=r"$\pm 1\sigma$ Spread")
    axs[0, 0].plot(steps, r_max, color="#2ca02c", linestyle="--", alpha=0.7, label="Max Reward (Optimal Expert)")
    axs[0, 0].plot(steps, r_min, color="#d62728", linestyle=":", alpha=0.7, label="Min Reward (Mismatched Expert)")
    axs[0, 0].set_xlabel("Steps")
    axs[0, 0].set_ylabel("Reward: -CER + eta*DialectBonus")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend(loc="lower right", fontsize=8)

    # Panel 2: Per-Expert CER & Dialect Classification Accuracy
    axs[0, 1].set_title("2. Per-Expert CER (%) & Dialect Alignment Acc", fontweight="bold", fontsize=11)
    ax2_sub = axs[0, 1].twinx()
    p1 = axs[0, 1].plot(steps, cer_e0, color="#1f77b4", linewidth=1.5, label="E0 Malvani CER")
    p2 = axs[0, 1].plot(steps, cer_e1, color="#ff7f0e", linewidth=1.5, label="E1 Ahirani CER")
    p3 = axs[0, 1].plot(steps, cer_e2, color="#2ca02c", linewidth=1.5, label="E2 Varhadi CER")
    p4 = ax2_sub.plot(steps, d_acc, color="#d62728", linewidth=2.0, linestyle="--", label="Dialect Choice Acc (%)")
    axs[0, 1].set_xlabel("Steps")
    axs[0, 1].set_ylabel("CER (%)")
    ax2_sub.set_ylabel("Dialect Match Acc (%)", color="#d62728")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    lines = p1 + p2 + p3 + p4
    labels = [l.get_label() for l in lines]
    axs[0, 1].legend(lines, labels, loc="upper right", fontsize=8)

    # Panel 3: Router Expert Load Balancing
    axs[1, 0].set_title("3. Expert Routing Load Distribution (%)", fontweight="bold", fontsize=11)
    axs[1, 0].plot(steps, load_e0, color="#1f77b4", linewidth=1.8, label="Expert 0: Malvani")
    axs[1, 0].plot(steps, load_e1, color="#ff7f0e", linewidth=1.8, label="Expert 1: Ahirani")
    axs[1, 0].plot(steps, load_e2, color="#2ca02c", linewidth=1.8, label="Expert 2: Varhadi")
    axs[1, 0].axhline(y=33.33, color="gray", linestyle=":", alpha=0.6, label="Ideal Equal Split (33.3%)")
    axs[1, 0].set_xlabel("Steps")
    axs[1, 0].set_ylabel("Selected Probability / Load (%)")
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend(loc="upper right", fontsize=8)

    # Panel 4: Loss Breakdown & Entropy
    axs[1, 1].set_title("4. Loss Dynamics & Router Entropy", fontweight="bold", fontsize=11)
    ax4_sub = axs[1, 1].twinx()
    p1 = axs[1, 1].plot(steps, l_tot, color="#333333", linewidth=2.0, label="Total Loss")
    p2 = axs[1, 1].plot(steps, l_surr, color="#e377c2", linewidth=1.5, label="GRPO Router Loss")
    p3 = axs[1, 1].plot(steps, l_ctc, color="#bcbd22", linewidth=1.2, linestyle="--", label="Supervised CTC Anchor")
    p4 = ax4_sub.plot(steps, r_ent, color="#8c564b", linewidth=1.8, linestyle="-.", label="Router Entropy")
    axs[1, 1].set_xlabel("Steps")
    axs[1, 1].set_ylabel("Loss Magnitude")
    ax4_sub.set_ylabel("Router Entropy", color="#8c564b")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    lines = p1 + p2 + p3 + p4
    labels = [l.get_label() for l in lines]
    axs[1, 1].legend(lines, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close(fig)


def train_moe_grpo(
    run_dir: str = "runs/run_moe_grpo",
    moe_ckpt: str = "runs/no-splitformer-moe-only/checkpoints/stage2_moe/best_model.pt",
    total_steps: int = 2500,
    batch_size: int = 4,
    gamma_wer: float = 0.15,
    eta_dialect_bonus: float = 0.08,
    clip_eps: float = 0.20,
    entropy_weight: float = 0.02,
    alpha_ctc: float = 0.20,
    router_lr: float = 1e-4,
    expert_lr: float = 2e-5,
    min_lr_ratio: float = 0.10,
    warmup_steps: int = 150,
    grad_accum_steps: int = 2,
    log_every: int = 10,
    plot_every: int = 25,
    save_every: int = 500,
    checkpoint_dir: Optional[str] = None,
    stage_name: str = "stage4b_moe_grpo",
    resume_path: Optional[str] = None,
):
    """
    Main training routine for Stage 4B GRPO Dialect Expert Routing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"[{stage_name}] MoE Dialect Expert Routing GRPO on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Configuration:")
    print(f"    - Base Checkpoint       : {moe_ckpt}")
    print(f"    - Steps                 : {total_steps}")
    print(f"    - Batch Size            : {batch_size} (Grad Accum: {grad_accum_steps}, Effective: {batch_size * grad_accum_steps})")
    print(f"    - Group Size (G)        : 3 experts [E0: Malvani, E1: Ahirani, E2: Varhadi]")
    print(f"    - Reward Weights        : R = - (CER + {gamma_wer:.2f}*WER) + {eta_dialect_bonus:.2f}*I(expert == dialect)")
    print(f"    - Policy Clipping       : eps = {clip_eps:.2f}")
    print(f"    - Router Entropy Weight : {entropy_weight:.4f}")
    print(f"    - CTC Anchor Weight     : {alpha_ctc:.4f}")
    print(f"    - Router LR / Expert LR : {router_lr:.2e} / {expert_lr:.2e}")
    print("=" * 80 + "\n")

    actual_run_dir = Path(run_dir)
    actual_ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else actual_run_dir / "checkpoints" / stage_name
    plots_dir = actual_run_dir / "plots"
    actual_ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    print(f"[Tokenizer] Loaded vocabulary with {tokenizer.vocab_size} tokens")

    # 1. Load MoE Model
    print(f"[Model] Initializing MoE Conformer from: {moe_ckpt}")
    model = load_moe_model(moe_ckpt, device=device)

    # 2. Identify Routers vs Experts vs Backbone Parameters
    router_params = []
    expert_params = []
    backbone_params = []

    # Freeze acoustic backbone blocks 1-4
    for layer_idx in range(4):
        for param in model.encoder.layers[layer_idx].parameters():
            param.requires_grad = False
    for param in model.encoder.subsampling.parameters():
        param.requires_grad = False
    for param in model.encoder.pos_enc.parameters():
        param.requires_grad = False
    for param in model.frontend.parameters():
        param.requires_grad = False

    # In layers 5-12, separate router parameters and expert parameters
    moe_layers_list = []
    for layer_idx in range(4, 12):
        block = model.encoder.layers[layer_idx]
        if type(block.ffn2).__name__ == "SparseMoELayer":
            moe_layers_list.append(block.ffn2)
            # Router weights
            for param in block.ffn2.router.parameters():
                param.requires_grad = True
                router_params.append(param)
            # Expert networks + Combining shared FFN
            for param in block.ffn2.experts.parameters():
                param.requires_grad = True
                expert_params.append(param)
            for param in block.ffn2.combining_ffn.parameters():
                param.requires_grad = True
                expert_params.append(param)

    # CTC projection head (trainable at expert_lr)
    for param in model.ctc_head.parameters():
        param.requires_grad = True
        expert_params.append(param)

    print(f"[MoE Architecture] Found {len(moe_layers_list)} SparseMoELayers (Layers 5–12).")
    print(f"  - Router Parameters (RL Policy)   : {sum(p.numel() for p in router_params):,} params")
    print(f"  - Expert + CTC Head Parameters    : {sum(p.numel() for p in expert_params):,} params")

    # 3. Load Kathbath Dataset
    print("[Data] Ingesting Kathbath manifest for Dialect MoE Routing...")
    kathbath_manifest = "data/vistaar_benchmarks/kathbath_train/marathi/manifest.json"
    kathbath_dir = "data/vistaar_benchmarks/kathbath_train/marathi"
    train_utts = load_kathbath_manifest(kathbath_manifest, kathbath_dir)
    total_hours = sum(u["duration"] for u in train_utts) / 3600.0
    print(f"[Data] Kathbath dataset verified: {len(train_utts):,} utterances ({total_hours:.1f} hours)")

    loader = LocalKathbathPrefetchLoader(train_utts, tokenizer, batch_size=batch_size, queue_size=4)

    # 4. Optimizer Setup: Dual learning rate for router vs experts
    optimizer = AdamW([
        {"params": router_params, "lr": router_lr},
        {"params": expert_params, "lr": expert_lr},
    ], weight_decay=1e-4, betas=(0.9, 0.98))

    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio)
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

    print(f"\n[{stage_name}] Commencing MoE Dialect Routing GRPO (Target Steps: {total_steps})...\n")

    try:
        while step < total_steps:
            step_start = time.time()
            batch = next(loader)
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = (batch.get("audio_lens") if "audio_lens" in batch else batch["audio_lengths"]).to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lens = (batch.get("target_lens") if "target_lens" in batch else batch["target_lengths"]).to(device, non_blocking=True)
            gt_dialect_idx = batch["dialect_indices"].to(device, non_blocking=True)
            ref_texts = batch["texts"]

            b_curr = audio.size(0)
            step += 1

            # -----------------------------------------------------------------
            # Step A: Evaluate G=3 Rollouts (Force-routing to Expert 0, 1, 2)
            # -----------------------------------------------------------------
            expert_hyp_texts = {0: [], 1: [], 2: []}
            expert_cers = {0: [], 1: [], 2: []}
            expert_wers = {0: [], 1: [], 2: []}
            expert_rewards = {0: [], 1: [], 2: []}

            # We evaluate each expert with torch.no_grad() for reward determination
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    for exp_k in range(3):
                        forced_dialect = torch.full((b_curr,), exp_k, dtype=torch.long, device=device)
                        ctc_dict_k = model.forward_ctc(audio, dialect_idx=forced_dialect, use_hard_routing=True)
                        log_probs_k = ctc_dict_k["log_probs"]  # (B, T, V)
                        t_k = log_probs_k.size(1)
                        valid_lens_k = torch.clamp((audio_lens // 160 // 4), max=t_k)

                        for b_i in range(b_curr):
                            v_len = int(valid_lens_k[b_i].item())
                            path_k = torch.argmax(log_probs_k[b_i, :v_len], dim=-1).tolist()
                            collapsed = collapse_ctc_path(path_k, blank_id=tokenizer.blank_id)
                            hyp_k = tokenizer.ctc_decode(collapsed)
                            ref_t = ref_texts[b_i]

                            cer_val = compute_cer(ref_t, hyp_k)
                            wer_val = compute_wer(ref_t, hyp_k)

                            # Dialect bonus: add reward if expert matches ground-truth dialect
                            dialect_match = 1.0 if (int(gt_dialect_idx[b_i].item()) == exp_k) else 0.0
                            reward_val = - (cer_val + gamma_wer * wer_val) + (eta_dialect_bonus * dialect_match)

                            expert_hyp_texts[exp_k].append(hyp_k)
                            expert_cers[exp_k].append(cer_val)
                            expert_wers[exp_k].append(wer_val)
                            expert_rewards[exp_k].append(reward_val)

            # -----------------------------------------------------------------
            # Step B: Normalized Relative Group Advantage (GRPO)
            # -----------------------------------------------------------------
            # Shape: (B, 3)
            rewards_mat = torch.tensor([
                [expert_rewards[0][b_i], expert_rewards[1][b_i], expert_rewards[2][b_i]]
                for b_i in range(b_curr)
            ], dtype=torch.float32, device=device)

            group_mean = rewards_mat.mean(dim=-1, keepdim=True)
            group_std = rewards_mat.std(dim=-1, keepdim=True) + 1e-8
            advantages = (rewards_mat - group_mean) / group_std  # (B, 3)

            # -----------------------------------------------------------------
            # Step C: Dynamic Routing Forward Pass (Trainable Router Policy)
            # -----------------------------------------------------------------
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # Pass through model in dynamic soft/top-1 routing mode (use_hard_routing=False)
                dyn_ctc_dict = model.forward_ctc(audio, dialect_idx=None, use_hard_routing=False)
                dyn_log_probs = dyn_ctc_dict["log_probs"]  # (B, T, V)
                t_frames = dyn_log_probs.size(1)
                valid_lens = torch.clamp((audio_lens // 160 // 4), max=t_frames)

                # Extract router distributions from all MoE layers
                layer_router_probs = []
                for moe_mod in moe_layers_list:
                    # moe_mod.current_routing_stats contains per-expert densities
                    # moe_mod.router is nn.Linear(d_model, num_experts)
                    # We pool router probabilities across frames
                    # To obtain differentiable log-probabilities per utterance:
                    # In SparseMoELayer forward: router_logits was computed
                    pass

                # Compute router probabilities across MoE layers
                total_surrogate_loss = torch.tensor(0.0, device=device)
                total_entropy = torch.tensor(0.0, device=device)
                clip_count = 0
                total_actions = 0

                # Loop over MoE layers to compute policy gradient on router
                for moe_mod in moe_layers_list:
                    # Recompute router logits on the layer's cached input or compute utterance policy
                    # In our architecture, router weights take hidden representation
                    # To ensure end-to-end gradient flow directly into the router:
                    # We compute the router's softmax probability over the 3 experts
                    r_weight = moe_mod.router.weight  # (3, d_model)
                    r_bias = moe_mod.router.bias      # (3,)

                    # Use final_hidden as an anchor to extract router policy distribution per sample
                    # Layer router logits: (B, 3)
                    pooled_hidden = dyn_ctc_dict["final_hidden"].mean(dim=1)  # (B, d_model)
                    router_logits = F.linear(pooled_hidden, r_weight, r_bias).float()  # (B, 3) in fp32
                    router_p = F.softmax(router_logits, dim=-1)                        # (B, 3)
                    router_log_p = F.log_softmax(router_logits, dim=-1)                # (B, 3)
                    layer_router_probs.append(router_p.detach())

                    # Policy entropy: - sum(p * log_p)
                    ent = - (router_p * router_log_p).sum(dim=-1).mean()
                    total_entropy = total_entropy + ent

                    # Behavior policy (detached reference)
                    old_router_log_p = router_log_p.detach()

                    for exp_k in range(3):
                        adv_k = advantages[:, exp_k]  # (B,)
                        # Ratio: r_k(psi) = exp(log_p - log_p_old)
                        ratio_k = torch.exp(router_log_p[:, exp_k] - old_router_log_p[:, exp_k])

                        surr1 = ratio_k * adv_k
                        surr2 = torch.clamp(ratio_k, 1.0 - clip_eps, 1.0 + clip_eps) * adv_k
                        surr_k = - torch.min(surr1, surr2).mean()

                        total_surrogate_loss = total_surrogate_loss + surr_k

                        with torch.no_grad():
                            clipped = (torch.abs(ratio_k - 1.0) > clip_eps).sum().item()
                            clip_count += clipped
                            total_actions += b_curr

                num_moe = max(len(moe_layers_list), 1)
                loss_router_surr = total_surrogate_loss / (num_moe * 3.0)
                mean_entropy = total_entropy / num_moe

                # -------------------------------------------------------------
                # Step D: Auxiliary Supervised CTC Loss on Output
                # -------------------------------------------------------------
                dyn_log_probs_t = dyn_log_probs.transpose(0, 1)  # (T, B, V)
                loss_ctc = ctc_criterion(dyn_log_probs_t, targets, valid_lens, target_lens)

                # -------------------------------------------------------------
                # Step E: Total Objective
                # -------------------------------------------------------------
                total_loss = loss_router_surr + (alpha_ctc * loss_ctc) - (entropy_weight * mean_entropy)

            # Backward Pass
            loss_scaled = total_loss / grad_accum_steps
            scaler.scale(loss_scaled).backward()

            grad_norm_val = 0.0
            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(router_params + expert_params, max_norm=5.0)
                grad_norm_val = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # -----------------------------------------------------------------
            # Step F: Collect Telemetry Metrics
            # -----------------------------------------------------------------
            step_time = time.time() - step_start
            batch_audio_sec = float(audio_lens.sum().item()) / 16000.0
            throughput = batch_audio_sec / max(step_time, 1e-5)

            with torch.no_grad():
                # Router load distribution across batch
                avg_layer_probs = torch.stack(layer_router_probs, dim=0).mean(dim=0)  # (B, 3)
                load_dist = avg_layer_probs.mean(dim=0).tolist()                      # [e0, e1, e2]

                # Dialect prediction accuracy
                chosen_experts = avg_layer_probs.argmax(dim=-1)                       # (B,)
                dialect_matches = (chosen_experts == gt_dialect_idx).float().mean().item()

                # Dynamic decode CER
                dyn_cers = []
                for b_i in range(b_curr):
                    v_len = int(valid_lens[b_i].item())
                    dyn_p = torch.argmax(dyn_log_probs[b_i, :v_len], dim=-1).tolist()
                    dyn_hyp = tokenizer.ctc_decode(collapse_ctc_path(dyn_p, blank_id=tokenizer.blank_id))
                    dyn_cers.append(compute_cer(ref_texts[b_i], dyn_hyp))
                mean_dyn_cer = float(np.mean(dyn_cers))

                # Aggregate reward statistics
                all_rewards = rewards_mat.view(-1).tolist()

                step_metrics = {
                    # Rewards
                    "reward_mean": round(float(np.mean(all_rewards)), 4),
                    "reward_std": round(float(np.mean([np.std(rewards_mat[b_i].tolist()) for b_i in range(b_curr)])), 4),
                    "reward_min": round(float(np.min(all_rewards)), 4),
                    "reward_max": round(float(np.max(all_rewards)), 4),
                    # Error Rates
                    "dynamic_cer": round(mean_dyn_cer, 4),
                    "expert_cer_e0": round(float(np.mean(expert_cers[0])), 4),
                    "expert_cer_e1": round(float(np.mean(expert_cers[1])), 4),
                    "expert_cer_e2": round(float(np.mean(expert_cers[2])), 4),
                    "oracle_expert_cer": round(float(np.mean([min(expert_cers[0][i], expert_cers[1][i], expert_cers[2][i]) for i in range(b_curr)])), 4),
                    # Router Loads
                    "router_load_e0": round(float(load_dist[0]), 4),
                    "router_load_e1": round(float(load_dist[1]), 4),
                    "router_load_e2": round(float(load_dist[2]), 4),
                    "dialect_acc": round(float(dialect_matches), 4),
                    # Losses & Diagnostics
                    "loss_total": round(float(total_loss.item()), 4),
                    "loss_grpo_surr": round(float(loss_router_surr.item()), 4),
                    "loss_ctc": round(float(loss_ctc.item()), 4),
                    "router_entropy": round(float(mean_entropy.item()), 3),
                    "clip_fraction": round(float(clip_count / max(total_actions, 1)), 4),
                    "advantage_spread": round(float(advantages.max() - advantages.min()), 3),
                    "grad_norm": round(grad_norm_val, 3),
                    "lr_router": float(scheduler.get_last_lr()[0]),
                    "throughput_aud_s_per_s": round(throughput, 1),
                    "step_time_sec": round(step_time, 2),
                }

            # Telemetry logging
            logger.log_step(step, step_metrics)

            # Terminal Display
            if step % log_every == 0 or step == 1:
                print(
                    f"Step {step:5d}/{total_steps} | MoE GRPO | "
                    f"R_mean: {step_metrics['reward_mean']:+.3f} (std: {step_metrics['reward_std']:.3f}) | "
                    f"Dyn CER: {step_metrics['dynamic_cer']*100:5.2f}% (Oracle: {step_metrics['oracle_expert_cer']*100:5.2f}%) | "
                    f"E0/E1/E2 CER: {step_metrics['expert_cer_e0']*100:.1f}%/{step_metrics['expert_cer_e1']*100:.1f}%/{step_metrics['expert_cer_e2']*100:.1f}% | "
                    f"Loads: [{step_metrics['router_load_e0']*100:2.0f}%, {step_metrics['router_load_e1']*100:2.0f}%, {step_metrics['router_load_e2']*100:2.0f}%] | "
                    f"Dialect Acc: {step_metrics['dialect_acc']*100:4.1f}% | "
                    f"Surr: {step_metrics['loss_grpo_surr']:+.3f} | CTC: {step_metrics['loss_ctc']:.3f} | Ent: {step_metrics['router_entropy']:.2f}"
                )
                ref_d = DIALECT_NAMES.get(int(gt_dialect_idx[0].item()), "Unknown")
                pred_d = DIALECT_NAMES.get(int(chosen_experts[0].item()), "Unknown")
                print(f"  [Dialect GT/Pred]: {ref_d} -> {pred_d}")
                print(f"  [Ref  ]: {ref_texts[0][:75]}")
                print(f"  [HypE0]: {expert_hyp_texts[0][0][:75]}")
                print(f"  [HypE1]: {expert_hyp_texts[1][0][:75]}")
                print(f"  [HypE2]: {expert_hyp_texts[2][0][:75]}\n")

            # Periodic Plot Generation
            if step % plot_every == 0 or step == 1:
                generate_moe_grpo_plots(logger.csv_path, dashboard_png, stage_name)

            # Checkpointing
            if step % save_every == 0 or step == total_steps:
                logger.save_checkpoint(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_val=step_metrics["dynamic_cer"],
                    metric_name="dynamic_cer",
                    lower_is_better=True,
                    metadata={
                        "dialect_acc": step_metrics["dialect_acc"],
                        "reward_mean": step_metrics["reward_mean"],
                        "oracle_expert_cer": step_metrics["oracle_expert_cer"],
                    },
                )
                print(f"  [Checkpoint] Step {step} saved to: {logger.ckpt_dir}")

    finally:
        loader.close()

    total_time = (time.time() - start_time) / 3600.0
    generate_moe_grpo_plots(logger.csv_path, dashboard_png, stage_name)
    print(f"\n[Completed] MoE Dialect Routing GRPO finished in {total_time:.2f} hours!")
    print(f"  Final Dashboard Plot: {dashboard_png.resolve()}")
    print(f"  Final Metrics CSV   : {logger.csv_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4B: MoE Dialect Routing via GRPO")
    parser.add_argument("--moe_ckpt", type=str, default="runs/no-splitformer-moe-only/checkpoints/stage2_moe/best_model.pt", help="MoE checkpoint to train")
    parser.add_argument("--steps", type=int, default=2500, help="Total GRPO steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--gamma_wer", type=float, default=0.15, help="WER penalty coefficient in reward")
    parser.add_argument("--eta_dialect_bonus", type=float, default=0.08, help="Bonus for selecting ground truth dialect")
    parser.add_argument("--clip_eps", type=float, default=0.20, help="PPO clipping epsilon")
    parser.add_argument("--entropy_weight", type=float, default=0.02, help="Router entropy regularization weight")
    parser.add_argument("--alpha_ctc", type=float, default=0.20, help="Supervised CTC anchor loss coefficient")
    parser.add_argument("--router_lr", type=float, default=1e-4, help="Router learning rate")
    parser.add_argument("--expert_lr", type=float, default=2e-5, help="Expert learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.10, help="Minimum LR ratio")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Warmup steps")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--log_every", type=int, default=10, help="Terminal log interval")
    parser.add_argument("--plot_every", type=int, default=25, help="Dashboard plot interval")
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--run_dir", type=str, default="runs/run_moe_grpo", help="Output run directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Custom checkpoint output directory")
    parser.add_argument("--stage_name", type=str, default="stage4b_moe_grpo", help="Stage identifier")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")

    args = parser.parse_args()

    train_moe_grpo(
        run_dir=args.run_dir,
        moe_ckpt=args.moe_ckpt,
        total_steps=args.steps,
        batch_size=args.batch_size,
        gamma_wer=args.gamma_wer,
        eta_dialect_bonus=args.eta_dialect_bonus,
        clip_eps=args.clip_eps,
        entropy_weight=args.entropy_weight,
        alpha_ctc=args.alpha_ctc,
        router_lr=args.router_lr,
        expert_lr=args.expert_lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        plot_every=args.plot_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        stage_name=args.stage_name,
        resume_path=args.resume,
    )
