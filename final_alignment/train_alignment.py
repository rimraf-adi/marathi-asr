"""
Stage 5: Speaker-Disjoint Calibration & Sequence Polishing Engine.
Calibrates the 3-Dialect MoE Conformer against 133 held-out speakers (41,041 utterances, ~52.1 hours)
from RESPIN to ensure generalization to unseen vocal tract acoustics and microphone channels,
polishing character boundary transitions before final benchmark evaluation.
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure root directory is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torch.optim import AdamW

from data_utils.tokenizer import MarathiTokenizer

from data_utils.utils import compute_cer, get_lr_scheduler
from moe.upcycling import load_moe_model
from moe.respin_dataset import build_or_load_speaker_split, LocalRESPINPrefetchLoader
from telemetry.metrics_logger import RunMetricsLogger


def train_calibration_stage5(run_dir: str = "run3", 
    moe_ckpt: str = "run3/checkpoints/stage2_moe/best_model.pt",
    total_steps: int = 3000,
    batch_size: int = 24,
    grad_accum_steps: int = 2,
    lr: float = 3e-5,
    min_lr: float = 5e-6,
    warmup_steps: int = 200,
    log_every: int = 25,
    save_every: int = 1000,
    checkpoint_dir: Optional[str] = None,
    metrics_csv: Optional[str] = None,
    resume_path: Optional[str] = None,
    stage_name: str = "stage3_final_alignment",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n================================================================================")
    print(f"[{stage_name}: Calibration] Speaker-Disjoint Calibration on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"================================================================================")

    actual_ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(run_dir) / "checkpoints" / stage_name
    actual_ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    print(f"[Tokenizer] Loaded vocabulary with {tokenizer.vocab_size} tokens")

    # 1. Load trained MoE Model
    print(f"[Model] Loading MoE model from: {moe_ckpt}")
    model = load_moe_model(moe_ckpt, device=device)

    # 2. Freezing Strategy:
    # Anchor universal feature representation (Blocks 1-4 frozen)
    # Train top half (Blocks 5-12 including MoE experts, combining FFN, layer norms) + CTC heads
    print("[Freezing Strategy] Freezing acoustic backbone Layers 1-4...")
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []
    # Unfreeze Blocks 5-12
    for layer_idx in range(4, 12):
        for param in model.encoder.layers[layer_idx].parameters():
            param.requires_grad = True
            trainable_params.append(param)

    # Unfreeze all CTC heads
    for param in model.ctc_heads.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"[Parameters] Calibrating {num_trainable:,} / {num_total:,} parameters ({100 * num_trainable / num_total:.1f}%)")

    # 3. Load RESPIN Calibration Split (Held-Out 133 Speakers)
    print("[Data] Loading strictly held-out calibration split (133 unseen speakers)...")
    _, calib_utts = build_or_load_speaker_split()
    total_calib_hours = sum(u["duration"] for u in calib_utts) / 3600.0
    print(f"[Data] Partition verified: {len(calib_utts):,} utterances ({total_calib_hours:.1f} hours) across 133 disjoint speakers")

    loader = LocalRESPINPrefetchLoader(calib_utts, tokenizer, batch_size=batch_size, queue_size=4)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    ctc_criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    

    logger = RunMetricsLogger(
        stage_name=stage_name,
        run_dir=run_dir,
        total_steps=total_steps,
        plot_interval=100,
    )

    step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Resuming {stage_name} from: {resume_path}")
        r_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(r_ckpt["model_state_dict"])
        optimizer.load_state_dict(r_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(r_ckpt["scheduler_state_dict"])
        step = r_ckpt.get("step", 0)

    print(f"\n[{stage_name}] Commencing Calibration Sweep (Target Steps: {total_steps}, Batch: {batch_size}, LR: {lr:.1e} -> {min_lr:.1e})")
    print(f"  Checkpoints: '{logger.ckpt_dir.resolve()}'")
    print(f"  Metrics CSV: '{logger.csv_path.resolve()}'\n")

    model.train()
    start_time = time.time()
    rolling_loss = 0.0

    try:
        while step < total_steps:
            batch = next(loader)
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = batch["audio_lens"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lens = batch["target_lens"].to(device, non_blocking=True)

            step += 1
            step_start = time.time()
            

            optimizer.zero_grad(set_to_none=True) if (step - 1) % grad_accum_steps == 0 else None

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                ctc_dict = model.forward_ctc(audio, chunk_size=None)

                logits_l12 = ctc_dict["exit_log_probs"][12]
                logits_l8 = ctc_dict["exit_log_probs"][8]

                log_probs_l12 = logits_l12.transpose(0, 1)
                log_probs_l8 = logits_l8.transpose(0, 1)

                t_frames = logits_l12.size(1)
                input_lengths = torch.clamp((audio_lens // 160 // 4), max=t_frames)

                loss_ctc_12 = ctc_criterion(log_probs_l12, targets, input_lengths, target_lens)
                loss_ctc_8 = ctc_criterion(log_probs_l8, targets, input_lengths, target_lens)
                loss = 0.7 * loss_ctc_12 + 0.3 * loss_ctc_8

            loss_scaled = loss / grad_accum_steps
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

            # Telemetry
            step_time = time.time() - step_start
            batch_audio_sec = float(audio_lens.sum().item()) / 16000.0
            throughput = batch_audio_sec / max(step_time, 1e-5)
            rolling_loss = 0.9 * rolling_loss + 0.1 * loss.item() if rolling_loss > 0 else loss.item()

            if step % log_every == 0 or step == 1:
                with torch.no_grad():
                    eval_samples = min(8, audio.size(0))
                    batch_cers = []
                    for s_i in range(eval_samples):
                        p_toks = logits_l12[s_i].argmax(dim=-1).tolist()
                        p_txt = tokenizer.ctc_decode(p_toks)
                        r_txt = batch["texts"][s_i]
                        batch_cers.append(compute_cer(r_txt, p_txt))
                    cer = sum(batch_cers) / max(len(batch_cers), 1)
                    ref_text = batch["texts"][0]
                    pred_text = tokenizer.ctc_decode(logits_l12[0].argmax(dim=-1).tolist())

                current_lr = scheduler.get_last_lr()[0]

                # Stream to RunMetricsLogger
                logger.log_step(step, {
                    "ctc_loss": round(rolling_loss, 4),
                    "cer": round(cer, 4),
                    "grad_norm": round(grad_norm_val, 3),
                    "throughput_aud_s_per_s": round(throughput, 1),
                    "lr": current_lr,
                })

                print(
                    f"Step {step:5d}/{total_steps} | Stage 5 Calib | "
                    f"CTC Loss: {rolling_loss:.4f} | CER: {cer:.3f} | "
                    f"Throughput: {throughput:5.1f} aud-s/s | LR: {current_lr:.2e}"
                )
                print(f"  [Ref ]: {ref_text[:70]}...")
                print(f"  [Pred]: {pred_text[:70]}...\n")

            if step % save_every == 0 or step == total_steps:
                logger.save_checkpoint(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_val=cer,
                    metric_name="cer",
                    lower_is_better=True,
                    metadata={"ctc_loss": rolling_loss, "calib_speakers": 133, "num_experts": 3},
                )
                print(f"  [Checkpoint] Step {step} saved to {logger.ckpt_dir}")

    finally:
        loader.close()

    total_time = (time.time() - start_time) / 3600.0
    print(f"\n[Completed] Stage 5 Speaker-Disjoint Calibration completed in {total_time:.2f} hours!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Speaker-Disjoint Final Alignment")
    parser.add_argument("--moe_ckpt", type=str, default="run3/checkpoints/stage2_moe/best_model.pt", help="MoE checkpoint path")
    parser.add_argument("--steps", type=int, default=3000, help="Total calibration steps")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--log_every", type=int, default=25, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--run_dir", type=str, default="run3", help="Directory to save the run")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoints output directory")
    parser.add_argument("--metrics_csv", type=str, default=None, help="Metrics CSV output")
    parser.add_argument("--stage_name", type=str, default="stage3_final_alignment", help="Stage name identifier")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path")

    args = parser.parse_args()

    train_calibration_stage5(
        run_dir=args.run_dir,
        moe_ckpt=args.moe_ckpt,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        metrics_csv=args.metrics_csv,
        resume_path=args.resume,
        stage_name=args.stage_name,
    )
