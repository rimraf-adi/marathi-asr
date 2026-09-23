"""
Stage 4: 3-Dialect Mixture-of-Experts (MoE) Dialect Adaptation Engine.
Trains the 3 Dialect Experts (Malvani, Ahirani, Varhadi) on RESPIN multi-dialect Marathi speech:
  - Phase 4A (Steps 1 - 3,000): Metadata-guided hard expert supervision
  - Phase 4B (Steps 3,001 - 10,000): Dynamic autonomous routing + load balancing
Semantically saves checkpoints to 'checkpoints/moe/'.
"""

import os
import sys
import time
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
from moe.moe_layer import SparseMoELayer
from moe.upcycling import upcycle_conformer_to_moe
from moe.respin_dataset import build_or_load_speaker_split, LocalRESPINPrefetchLoader
from telemetry.metrics_logger import RunMetricsLogger


def train_moe_stage4(run_dir: str = "run3", 
    pretrained_ckpt: str = "run3/checkpoints/stage1_pretrain/best_model.pt",
    total_steps: int = 10000,
    batch_size: int = 24,
    grad_accum_steps: int = 2,
    lr: float = 1e-4,
    warmup_steps: int = 1000,
    phase4a_steps: int = 3000,
    balance_loss_weight: float = 0.05,
    log_every: int = 25,
    save_every: int = 1000,
    checkpoint_dir: Optional[str] = None,
    resume_path: Optional[str] = None,
    stage_name: str = "stage2_moe",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{stage_name}: 3-Dialect MoE] Initializing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    actual_ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(run_dir) / "checkpoints" / stage_name
    actual_ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    print(f"[Tokenizer] Loaded vocabulary with {tokenizer.vocab_size} tokens")

    # 1. Sparse Upcycling from Pretrained Checkpoint
    model = upcycle_conformer_to_moe(
        pretrained_checkpoint_path=pretrained_ckpt,
        num_experts=3,
        moe_layers=[4, 5, 6, 7, 8, 9, 10, 11],  # Blocks 5 to 12
        device=device,
    )

    # 2. Freeze lower universal feature layers (1-4) and Combining Shared FFN to preserve standard Marathi
    print("[Stage 4] Freezing universal backbone (Layers 1-4) & Combining Shared FFN trunks...")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze ONLY the 3 Dialect Experts and Gating Routers in Layers 5-12 + CTC heads
    trainable_params = []
    for layer_idx in [4, 5, 6, 7, 8, 9, 10, 11]:
        block = model.encoder.layers[layer_idx]
        if isinstance(block.ffn2, SparseMoELayer):
            # Unfreeze experts
            for param in block.ffn2.experts.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            # Unfreeze router
            for param in block.ffn2.router.parameters():
                param.requires_grad = True
                trainable_params.append(param)

    # Unfreeze CTC projection heads
    for param in model.ctc_heads.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"[Parameters] Trainable Dialect MoE Parameters: {num_trainable:,} / {num_total:,} ({100 * num_trainable / num_total:.1f}%)")

    # 3. Load RESPIN Dataset Split
    train_utts, _ = build_or_load_speaker_split()
    loader = LocalRESPINPrefetchLoader(train_utts, tokenizer, batch_size=batch_size, queue_size=4)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
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

    print(f"\n[{stage_name}] Starting 3-Dialect MoE Training (Target Steps: {total_steps}, Batch Size: {batch_size})")
    print(f"  Phase 4A (Metadata-Guided Hard Routing): Steps 1 to {phase4a_steps}")
    print(f"  Phase 4B (Dynamic Soft/Top-1 Routing): Steps {phase4a_steps + 1} to {total_steps}")
    print(f"  Saving semantically to: '{actual_ckpt_dir.resolve()}'\n")

    model.train()
    start_time = time.time()
    rolling_ctc_loss = 0.0
    rolling_bal_loss = 0.0

    try:
        while step < total_steps:
            step += 1
            step_start = time.time()
            batch = next(loader)

            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = batch["audio_lens"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lens = batch["target_lens"].to(device, non_blocking=True)
            dialect_indices = batch["dialect_indices"].to(device, non_blocking=True)

            is_phase_4a = step <= phase4a_steps
            optimizer.zero_grad(set_to_none=True) if (step - 1) % grad_accum_steps == 0 else None

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                ctc_dict = model.forward_ctc(
                    audio,
                    dialect_idx=dialect_indices if is_phase_4a else None,
                    use_hard_routing=is_phase_4a,
                )

                input_lengths = ctc_dict["output_lengths"]
                logits_l12 = ctc_dict["exit_log_probs"][12]
                logits_l8 = ctc_dict["exit_log_probs"][8]

                log_probs_l12 = logits_l12.transpose(0, 1)
                log_probs_l8 = logits_l8.transpose(0, 1)

                loss_ctc_12 = ctc_criterion(log_probs_l12, targets, input_lengths, target_lens)
                loss_ctc_8 = ctc_criterion(log_probs_l8, targets, input_lengths, target_lens)
                ctc_loss = 0.7 * loss_ctc_12 + 0.3 * loss_ctc_8

                total_aux_loss = torch.tensor(0.0, device=device)
                for layer_idx in [4, 5, 6, 7, 8, 9, 10, 11]:
                    block = model.encoder.layers[layer_idx]
                    if isinstance(block.ffn2, SparseMoELayer) and hasattr(block.ffn2, "current_aux_loss"):
                        total_aux_loss = total_aux_loss + block.ffn2.current_aux_loss

                total_loss = ctc_loss + 0.01 * total_aux_loss

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

            step_time = time.time() - step_start
            batch_audio_sec = float(audio_lens.sum().item()) / 16000.0
            throughput = batch_audio_sec / max(step_time, 1e-5)

            rolling_ctc_loss = 0.9 * rolling_ctc_loss + 0.1 * ctc_loss.item() if rolling_ctc_loss > 0 else ctc_loss.item()
            aux_val = float(total_aux_loss.item()) if isinstance(total_aux_loss, torch.Tensor) else 0.0

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
                phase_name = "4A (Hard Routing)" if is_phase_4a else "4B (Dynamic MoE)"

                # Stream to RunMetricsLogger
                logger.log_step(step, {
                    "phase": phase_name,
                    "ctc_loss": round(rolling_ctc_loss, 4),
                    "aux_loss": round(aux_val, 4),
                    "cer": round(cer, 4),
                    "grad_norm": round(grad_norm_val, 3),
                    "throughput_aud_s_per_s": round(throughput, 1),
                    "lr": current_lr,
                })

                print(
                    f"Step {step:6d}/{total_steps} | Phase: {phase_name} | "
                    f"CTC Loss: {rolling_ctc_loss:.4f} | Aux Loss: {aux_val:.4f} | CER: {cer:.3f} | "
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
                    metadata={"ctc_loss": rolling_ctc_loss, "aux_loss": aux_val, "num_experts": 3},
                )
                print(f"  [Checkpoint] Step {step} saved to {logger.ckpt_dir}")

    finally:
        loader.close()

    print(f"\n[Completed] Stage 4 MoE Dialect Adaptation finished in {(time.time() - start_time)/3600.0:.2f} hours!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: 3-Dialect MoE Adaptation")
    parser.add_argument("--pretrained_ckpt", type=str, default="run3/checkpoints/stage1_pretrain/best_model.pt")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--phase4a_steps", type=int, default=3000, help="Phase 4A steps")
    parser.add_argument("--log_every", type=int, default=25, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--run_dir", type=str, default="run3", help="Directory to save the run")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoints directory")
    parser.add_argument("--stage_name", type=str, default="stage2_moe", help="Stage name identifier")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")

    args = parser.parse_args()

    train_moe_stage4(
        run_dir=args.run_dir,
        pretrained_ckpt=args.pretrained_ckpt,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        phase4a_steps=args.phase4a_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        stage_name=args.stage_name,
    )
