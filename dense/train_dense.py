"""
Dense Conformer-Medium Dialect Adaptation Engine (MoE-Free Baseline).
Trains a standard dense 12-layer Conformer (d_model=512) on pooled RESPIN multi-dialect Marathi speech:
  - Identical active parameter capacity and forward FLOPs as MoE top-1 routing
  - All 12 layers use standard dense Feed-Forward Networks (no MoE layers, no routing)
  - Differential LR: Layers 1-4 backbone at 1/10 LR, Layers 5-12 + CTC head at full LR
  - Pooled training without dialect partitioning to test the Negative Transfer hypothesis
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.optim import AdamW

from data_utils.tokenizer import MarathiTokenizer
from data_utils.utils import compute_cer, get_lr_scheduler
from model.asr_model import StreamingASRModel
from moe.respin_dataset import build_or_load_speaker_split, LocalRESPINPrefetchLoader
from telemetry.metrics_logger import RunMetricsLogger


def build_dense_conformer(
    pretrained_checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    d_model: int = 512,
    n_heads: int = 8,
    ffn_expansion: int = 8,
    num_layers: int = 12,
    vocab_size: int = 105,
) -> StreamingASRModel:
    """Instantiates a dense Conformer and loads pre-trained backbone weights."""
    print(f"\n[Dense Model] Initializing Dense Conformer (d_model={d_model}, n_heads={n_heads}, layers={num_layers}, ffn_exp={ffn_expansion})...")
    model = StreamingASRModel(
        feat_dim=80,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=n_heads,
        conv_kernel_size=31,
        ffn_expansion=ffn_expansion,
        dropout=0.1,
        exit_layers=[num_layers],
        enable_reconstruction_head=False,
        vocab_size=vocab_size,
    ).to(device)

    if pretrained_checkpoint_path and os.path.exists(pretrained_checkpoint_path):
        print(f"[Dense Model] Loading pre-trained weights from: {pretrained_checkpoint_path}")
        ckpt = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Filter out reconstruction head keys if present
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("reconstruction_head."):
                continue
            if k.startswith("backbone."):
                backbone_dict[k.replace("backbone.", "")] = v
            else:
                backbone_dict[k] = v

        missing, unexpected = model.load_state_dict(backbone_dict, strict=False)
        print(f"[Dense Model] Loaded weights successfully! Missing keys: {len(missing)} (e.g. CTC head), Unexpected keys: {len(unexpected)}")
    else:
        print("[Dense Model] Warning: No pre-trained checkpoint provided. Initializing from scratch.")

    return model


def train_dense_stage2(
    run_dir: str = "runs/scaled-conformer512-dense-3stage",
    pretrained_ckpt: str = "runs/scaled-conformer512-dense-3stage/checkpoints/stage1_pretrain/best_model.pt",
    total_steps: int = 80000,
    batch_size: int = 32,
    grad_accum_steps: int = 4,
    lr: float = 1.5e-4,
    warmup_steps: int = 3000,
    ctc_warmup_steps: int = 3000,
    log_every: int = 25,
    save_every: int = 1000,
    checkpoint_dir: Optional[str] = None,
    resume_path: Optional[str] = None,
    stage_name: str = "stage2_dense",
    d_model: int = 512,
    n_heads: int = 8,
    ffn_expansion: int = 8,
    num_layers: int = 12,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{stage_name}: Dense Baseline] Initializing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    actual_ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(run_dir) / "checkpoints" / stage_name
    actual_ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    print(f"[Tokenizer] Loaded vocabulary with {tokenizer.vocab_size} tokens")

    # 1. Instantiate pure Dense Conformer
    model = build_dense_conformer(
        pretrained_checkpoint_path=pretrained_ckpt,
        device=device,
        d_model=d_model,
        n_heads=n_heads,
        ffn_expansion=ffn_expansion,
        num_layers=num_layers,
        vocab_size=tokenizer.vocab_size,
    )

    # 2. Differential LR setup: Layers 1-4 backbone at 1/10 LR, Layers 5-12 + CTC head at full LR
    print("[Dense Training] Setting up differential LR — backbone (Layers 1-4) at 1/10 LR, Layers 5-12 + CTC at full LR...")
    backbone_params = []
    top_params = []

    for layer_idx in range(4):
        for param in model.encoder.layers[layer_idx].parameters():
            backbone_params.append(param)

    for layer_idx in range(4, 12):
        for param in model.encoder.layers[layer_idx].parameters():
            top_params.append(param)

    for param in model.ctc_head.parameters():
        top_params.append(param)

    for param in model.encoder.subsampling.parameters():
        backbone_params.append(param)
    for param in model.encoder.pos_enc.parameters():
        backbone_params.append(param)
    for param in model.frontend.parameters():
        param.requires_grad = False

    trainable_params = backbone_params + top_params
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"[Parameters] Total: {num_total:,} | Trainable: {num_trainable:,} ({100 * num_trainable / num_total:.1f}%)")
    print(f"  Backbone (1/10 LR): {sum(p.numel() for p in backbone_params):,} params")
    print(f"  Top Layers + CTC (full LR): {sum(p.numel() for p in top_params):,} params")

    # 3. Load RESPIN Dataset Split (Pooled Marathi Speech across all dialects)
    train_utts, _ = build_or_load_speaker_split()
    loader = LocalRESPINPrefetchLoader(train_utts, tokenizer, batch_size=batch_size, queue_size=4)

    optimizer = AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": top_params, "lr": lr},
    ], weight_decay=1e-4, betas=(0.9, 0.98))
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

    print(f"\n[{stage_name}] Starting Dense Conformer Adaptation (Target Steps: {total_steps}, Batch: {batch_size}, GradAccum: {grad_accum_steps})")
    print(f"  Saving semantically to: '{actual_ckpt_dir.resolve()}'\n")

    model.train()
    start_time = time.time()
    rolling_ctc_loss = 0.0

    sample_cer = 1.0
    avg_ctc = 10.0

    try:
        while step < total_steps:
            step += 1
            step_start = time.time()
            batch = next(loader)

            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = (batch.get("audio_lens") if "audio_lens" in batch else batch["audio_lengths"]).to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lens = (batch.get("target_lens") if "target_lens" in batch else batch["target_lengths"]).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) if (step - 1) % grad_accum_steps == 0 else None

            try:
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    ctc_dict = model.forward_ctc(audio)
                    input_lengths = ctc_dict["output_lengths"]
                    log_probs = ctc_dict["log_probs"]
                    log_probs_t = log_probs.transpose(0, 1)

                    ctc_loss = ctc_criterion(log_probs_t, targets, input_lengths, target_lens)
                    if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                        print(f"Warning: NaN/Inf loss at step {step}, skipping batch...")
                        continue

                    total_loss = ctc_loss / grad_accum_steps

                scaler.scale(total_loss).backward()

                if step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                rolling_ctc_loss += ctc_loss.item()

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] Out-of-memory at step {step}, skipping and clearing cache...")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            # Periodic Logging
            if step % log_every == 0 or step == 1:
                denom = max(1, log_every if step > 1 else 1)
                avg_ctc = rolling_ctc_loss / denom
                rolling_ctc_loss = 0.0

                elapsed = max(1e-5, time.time() - step_start)
                total_audio_sec = (audio_lens.sum().item() / 16000.0)
                throughput = total_audio_sec / elapsed

                with torch.no_grad():
                    eval_samples = min(8, audio.size(0))
                    batch_cers = []
                    for s_i in range(eval_samples):
                        p_toks = log_probs[s_i].argmax(dim=-1).tolist()
                        p_txt = tokenizer.ctc_decode(p_toks)
                        r_txt = batch["texts"][s_i]
                        batch_cers.append(compute_cer(r_txt, p_txt))
                    sample_cer = sum(batch_cers) / max(len(batch_cers), 1)
                    ref_text = batch["texts"][0]
                    pred_text = tokenizer.ctc_decode(log_probs[0].argmax(dim=-1).tolist())

                cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
                print(f"Step {step:6d}/{total_steps} | Dense Baseline | CTC Loss: {avg_ctc:.4f} | CER: {sample_cer:.3f} | Throughput: {throughput:.1f} aud-s/s | LR: {cur_lr:.2e}")
                print(f"  [Ref ]: {ref_text[:70]}...")
                print(f"  [Pred]: {pred_text[:70]}...\n")

                logger.log_step(
                    step=step,
                    metrics={
                        "ctc_loss": avg_ctc,
                        "cer": sample_cer,
                        "throughput_aud_s_per_s": throughput,
                        "lr": cur_lr,
                    },
                )

            # Periodic Checkpoint Saving
            if step % save_every == 0 or step == total_steps:
                logger.save_checkpoint(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_val=avg_ctc,
                    metric_name="ctc_loss",
                    lower_is_better=True,
                    keep_last_k=1,
                    milestone_step=0,
                    metadata={
                        "epoch": step,
                        "cer": sample_cer,
                        "model_config": {
                            "d_model": d_model,
                            "n_heads": n_heads,
                            "ffn_expansion": ffn_expansion,
                            "num_layers": num_layers,
                            "enable_moe": False,
                            "vocab_size": 105,
                        },
                    },
                )
                print(f"  [Checkpoint] Step {step} saved to {logger.ckpt_dir}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n[Dense Training] Interrupted by user.")
    finally:
        loader.close()
        total_time_min = (time.time() - start_time) / 60.0
        print(f"\n[{stage_name}] Finished after {total_time_min:.1f} minutes.")



def main():
    parser = argparse.ArgumentParser(description="Stage 2: Dense Conformer Dialect Adaptation")
    parser.add_argument("--run_dir", type=str, default="runs/scaled-conformer512-dense-3stage")
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--ctc_warmup_steps", type=int, default=3000)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stage_name", type=str, default="stage2_dense")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ffn_expansion", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=1000)
    args = parser.parse_args()

    train_dense_stage2(
        run_dir=args.run_dir,
        pretrained_ckpt=args.pretrained_ckpt,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        ctc_warmup_steps=args.ctc_warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        stage_name=args.stage_name,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ffn_expansion=args.ffn_expansion,
        num_layers=args.num_layers,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
