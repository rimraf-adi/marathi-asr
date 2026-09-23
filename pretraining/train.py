"""
Self-Supervised Pretraining Engine for Marathi & Multi-Dialect Speech.
Includes:
  - Epoch-based training (finite training for X epochs)
  - Small-chunk local disk caching with automatic network relay upon cache exhaustion
  - Hugging Face rate limit resilience
  - Phase 3 intrinsic gate checks, spectrogram plotting, and automated logging
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dotenv import load_dotenv

# Ensure root dir is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import StreamingASRModel
from pretraining.dataset_loader import get_combined_stream, CachedRelayedStream
from pretraining.prefetch_loader import ThreadedPrefetchLoader
from telemetry.metrics_logger import RunMetricsLogger
from pretraining.metrics import PretrainMetricsTracker

# Prevent Windows OpenMP / MKL thread contention and deadlock
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(4)  # Leave CPU threads free for background streaming prefetch


def get_timestamp() -> str:
    """Returns current timestamp in ISO format with milliseconds."""
    from datetime import datetime
    return datetime.now().isoformat(timespec="milliseconds")


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def collate_audio_batch(batch_items: List[Dict[str, Any]], target_sr: int = 16000) -> torch.Tensor:
    """
    Pads a list of variable-length audio arrays into a single batch tensor.
    Args:
        batch_items: List of dicts with key 'audio' (numpy array)
    Returns:
        waveforms: (batch_size, max_len) float32 tensor
    """
    tensors = [torch.from_numpy(item["audio"]).float() if isinstance(item["audio"], np.ndarray) else item["audio"].float() for item in batch_items]
    lengths = [t.size(0) for t in tensors]
    max_len = max(lengths)

    # Pad all to max_len
    padded = torch.zeros(len(tensors), max_len, dtype=torch.float32)
    for i, t in enumerate(tensors):
        padded[i, :t.size(0)] = t

    return padded


def run_intrinsic_evaluation_gate(
    model: StreamingASRModel,
    eval_waveforms: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Phase 3 Intrinsic Verification Gate:
      1. Short-mask (4 frames) vs. Long-mask (20 frames) comparison.
      2. Representation collapse check (effective rank).
    """
    model.eval()
    with torch.no_grad():
        eval_waveforms = eval_waveforms.to(device)

        # Test 1: Short masks (4 frames = ~40ms)
        out_short = model.forward_pretrain(eval_waveforms, custom_span_length=4, custom_mask_prob=0.35)
        loss_short = PretrainMetricsTracker.compute_reconstruction_metrics(
            out_short["original_spec"], out_short["reconstructed_spec"], out_short["mask"]
        )["loss_masked_l1"]

        # Test 2: Long masks (20 frames = ~200ms)
        out_long = model.forward_pretrain(eval_waveforms, custom_span_length=20, custom_mask_prob=0.35)
        loss_long = PretrainMetricsTracker.compute_reconstruction_metrics(
            out_long["original_spec"], out_long["reconstructed_spec"], out_long["mask"]
        )["loss_masked_l1"]

        degradation_ratio = (loss_long / (loss_short + 1e-8))
        rep_metrics = PretrainMetricsTracker.compute_representation_metrics(out_long["final_hidden"])

    model.train()
    return {
        "gate_short_mask_loss": loss_short,
        "gate_long_mask_loss": loss_long,
        "gate_mask_degradation_ratio": degradation_ratio,
        "gate_effective_rank": rep_metrics["effective_rank"],
        "gate_rank_passed": float(rep_metrics["effective_rank"] > 50.0),
        "gate_ratio_passed": float(degradation_ratio > 1.15),
    }


def train_pretrain(
    run_dir: str = "run3",
    epochs: int = 5,
    total_steps: Optional[int] = 30000,
    batch_size: int = 16,
    lr: float = 3e-4,
    warmup_steps: int = 1000,
    log_every: int = 20,
    plot_every: int = 200,
    eval_every: int = 500,
    save_every: int = 1000,
    exp_name: str = "marathi_ssl_pretrain",
    max_audio_sec: float = 12.0,
    min_audio_sec: float = 1.5,
    resume_path: Optional[str] = None,
    stage: str = "pretrain",
    checkpoint_dir: Optional[str] = None,
    cache_dir: str = "data_cache/pretrain",
    cache_chunk_size: int = 10000,
    max_cached_chunks: int = 10,
    # Early stopping
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-4,
    early_stop_metric: str = "masked_l1",
    # Stream factory override (used for unit/smoke tests)
    custom_stream_factory=None,
):
    """Main training loop with epoch bounds and small-chunk local caching + network relay."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{get_timestamp()} [Engine] Initializing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    actual_checkpoint_dir = checkpoint_dir or os.path.join(run_dir, "checkpoints", stage)
    os.makedirs(actual_checkpoint_dir, exist_ok=True)
    print(f"{get_timestamp()} [Engine] Semantic Checkpoint Directory: {actual_checkpoint_dir} (Stage: '{stage}')")

    effective_total_steps = total_steps if total_steps is not None else 50000
    tracker = PretrainMetricsTracker(log_dir=run_dir)
    logger = RunMetricsLogger(stage_name="stage1_pretrain", run_dir=run_dir, total_steps=effective_total_steps, plot_interval=100)

    # Initialize Streaming Conformer Model
    model = StreamingASRModel(
        feat_dim=80,
        d_model=256,
        num_layers=12,
        n_heads=4,
        conv_kernel_size=31,
        ffn_expansion=4,
        dropout=0.1,
        exit_layers=[4, 8, 12],
        enable_reconstruction_head=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{get_timestamp()} [Model] Conformer Backbone parameters: {total_params:,} ({total_params / 1e6:.2f} M)")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=effective_total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    step = 0
    start_epoch = 1
    total_audio_sec_processed = 0.0

    # Early stopping state
    best_metric_val = float("inf")
    early_stop_counter = 0
    early_stop_triggered = False

    # Resume from checkpoint if specified
    if resume_path and os.path.exists(resume_path):
        print(f"{get_timestamp()} [Resume] Loading checkpoint from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 1)
        total_audio_sec_processed = ckpt.get("total_audio_hours", 0.0) * 3600.0
        print(f"{get_timestamp()} [Resume] Resumed from epoch {start_epoch}, step {step} ({total_audio_sec_processed/3600.0:.2f} hrs processed)")

    # Initialize Cached Relayed Dataset
    print(f"{get_timestamp()} [Data] Initializing CachedRelayedStream (Cache Dir: '{cache_dir}', Chunk: {cache_chunk_size} samples)...")
    base_stream_factory = custom_stream_factory if custom_stream_factory is not None else (lambda: get_combined_stream(include_shrutilipi=True, include_vaani=True))
    cached_dataset = CachedRelayedStream(
        stream_factory=base_stream_factory,
        cache_dir=cache_dir,
        chunk_size=cache_chunk_size,
        max_cached_chunks=max_cached_chunks,
        min_duration=min_audio_sec,
        max_duration=max_audio_sec,
    )

    eval_buffer: List[Dict[str, Any]] = []
    recon_metrics = {"loss_masked_l1": 0.0, "reconstruction_snr_db": 0.0}

    print(f"{get_timestamp()} [Training] Beginning pretraining run '{exp_name}' for {epochs} epochs (Max Steps Cap: {total_steps})...\n")

    for current_epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*75}\n{get_timestamp()} [Epoch {current_epoch}/{epochs}] Commencing Training Pass\n{'='*75}\n")
        epoch_stream = cached_dataset.iter_epoch(current_epoch)
        prefetcher = ThreadedPrefetchLoader(
            stream_iterator=epoch_stream,
            batch_size=batch_size,
            max_prefetch=8,
            collate_fn=collate_audio_batch,
            device=device,
            min_duration=min_audio_sec,
            max_duration=max_audio_sec,
            infinite=False,
        )

        epoch_step = 0
        epoch_audio_sec = 0.0

        try:
            for waveforms, raw_items in prefetcher:
                if len(eval_buffer) < 16:
                    eval_buffer.extend(raw_items[: min(4, 16 - len(eval_buffer))])

                step += 1
                epoch_step += 1
                step_start = time.time()
                waveforms = waveforms.to(device)
                batch_audio_sec = sum(x["duration"] for x in raw_items)
                total_audio_sec_processed += batch_audio_sec
                epoch_audio_sec += batch_audio_sec

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model.forward_pretrain(waveforms)
                    orig = outputs["original_spec"]
                    recon = outputs["reconstructed_spec"]
                    mask = outputs["mask"]

                    mask_3d = mask.unsqueeze(-1).expand_as(orig)
                    if mask_3d.sum() > 0:
                        loss_l1 = torch.abs(orig[mask_3d] - recon[mask_3d]).mean()
                        loss_l2 = torch.mean((orig[mask_3d] - recon[mask_3d]) ** 2)
                        loss = loss_l1 + 0.5 * loss_l2
                    else:
                        loss = torch.tensor(0.0, device=orig.device, requires_grad=True)

                    recon_metrics = PretrainMetricsTracker.compute_reconstruction_metrics(orig, recon, mask)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                step_time = time.time() - step_start
                current_lr = scheduler.get_last_lr()[0]

                rep_metrics = PretrainMetricsTracker.compute_representation_metrics(outputs["final_hidden"])
                perf_metrics = {
                    "step_time_sec": step_time,
                    "throughput_audio_sec_per_sec": batch_audio_sec / max(1e-5, step_time),
                    "rtf_training": step_time / max(1e-5, batch_audio_sec),
                    "total_audio_hours": total_audio_sec_processed / 3600.0,
                    "gradient_norm": grad_norm,
                    "gpu_vram_allocated_gb": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
                    "gpu_vram_peak_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
                }

                logger.log_step(step, {
                    "epoch": current_epoch,
                    "masked_l1": round(recon_metrics["loss_masked_l1"], 4),
                    "snr_db": round(recon_metrics["reconstruction_snr_db"], 2),
                    "effective_rank": round(rep_metrics["effective_rank"], 1),
                    "rank_utilization_pct": round(rep_metrics["rank_utilization_pct"], 1),
                    "grad_norm": round(grad_norm, 2),
                    "throughput_aud_s_per_s": round(perf_metrics["throughput_audio_sec_per_sec"], 1),
                    "lr": current_lr,
                })

                if step % log_every == 0 or step == 1:
                    print(
                        f"{get_timestamp()} [E{current_epoch} Step {step:5d}] | "
                        f"Masked L1: {recon_metrics['loss_masked_l1']:.4f} | "
                        f"SNR: {recon_metrics['reconstruction_snr_db']:5.2f} dB | "
                        f"Eff. Rank: {rep_metrics['effective_rank']:5.1f} | "
                        f"GradNorm: {grad_norm:4.2f} | "
                        f"Throughput: {perf_metrics['throughput_audio_sec_per_sec']:5.1f} aud-s/s | "
                        f"LR: {current_lr:.2e}"
                    )

                if step % plot_every == 0 or step == 1:
                    plot_file = tracker.save_spectrogram_plot(
                        step=step,
                        original=orig,
                        masked=outputs["masked_spec"],
                        reconstructed=recon,
                        sample_idx=0,
                    )
                    print(f"{get_timestamp()}  [Artifact] Saved spectrogram plot -> {plot_file}")

                if step % eval_every == 0 and len(eval_buffer) >= 4:
                    eval_batch = collate_audio_batch(eval_buffer)
                    gate_results = run_intrinsic_evaluation_gate(model, eval_batch, device)
                    print(
                        f"{get_timestamp()} >>> [Phase 3 Gate Check @ Step {step}] "
                        f"Short Loss: {gate_results['gate_short_mask_loss']:.4f} | "
                        f"Long Loss: {gate_results['gate_long_mask_loss']:.4f} | "
                        f"Degradation Ratio: {gate_results['gate_mask_degradation_ratio']:.3f} | "
                        f"Gate Passed: {'YES' if gate_results['gate_ratio_passed'] and gate_results['gate_rank_passed'] else 'NO'}\n"
                    )

                if step % save_every == 0:
                    logger.save_checkpoint(
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metric_val=recon_metrics["loss_masked_l1"],
                        metric_name="masked_l1",
                        lower_is_better=True,
                        metadata={
                            "epoch": current_epoch,
                            "total_epochs": epochs,
                            "snr_db": recon_metrics["reconstruction_snr_db"],
                            "total_audio_hours": total_audio_sec_processed / 3600.0,
                        },
                    )
                    print(f"{get_timestamp()}  [Semantic Checkpoint] Saved to {logger.ckpt_dir}\n")

                    current_metric_val = recon_metrics.get(early_stop_metric, recon_metrics["loss_masked_l1"])
                    min_checkpoint_step = max(5, effective_total_steps // (save_every * 5))
                    if step >= min_checkpoint_step * save_every:
                        if current_metric_val < best_metric_val - early_stop_min_delta:
                            best_metric_val = current_metric_val
                            early_stop_counter = 0
                            print(f"{get_timestamp()}  [Early Stop] New best {early_stop_metric}: {best_metric_val:.6f}")
                        else:
                            early_stop_counter += 1
                            print(f"{get_timestamp()}  [Early Stop] No improvement ({early_stop_counter}/{early_stop_patience})")
                            if early_stop_counter >= early_stop_patience:
                                print(f"{get_timestamp()} [Early Stop] Triggered! Stopping training.")
                                early_stop_triggered = True

                if (total_steps is not None and step >= total_steps) or early_stop_triggered:
                    break

        finally:
            prefetcher.close()

        print(f"\n{get_timestamp()} [Epoch {current_epoch}/{epochs} Complete] Steps: {epoch_step:,} | Cumulative steps: {step:,} | Processed: {total_audio_sec_processed / 3600.0:.2f} hrs")

        # Save end-of-epoch checkpoint
        logger.save_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_val=recon_metrics["loss_masked_l1"],
            metric_name="masked_l1",
            lower_is_better=True,
            metadata={
                "epoch": current_epoch,
                "total_epochs": epochs,
                "snr_db": recon_metrics["reconstruction_snr_db"],
                "total_audio_hours": total_audio_sec_processed / 3600.0,
            },
        )

        if (total_steps is not None and step >= total_steps) or early_stop_triggered:
            print(f"{get_timestamp()} [Limit Reached] Stopping pretraining loop.")
            break

    print(f"\n{get_timestamp()} [Completed] Pretraining finished! Processed {total_audio_sec_processed / 3600.0:.2f} hours of speech across {epochs} epochs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marathi Conformer Self-Supervised Pretraining")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=None, help="Optional max training steps cap")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--log_every", type=int, default=10, help="Steps between log outputs")
    parser.add_argument("--plot_every", type=int, default=100, help="Steps between spectrogram plots")
    parser.add_argument("--eval_every", type=int, default=250, help="Steps between intrinsic gate evaluations")
    parser.add_argument("--save_every", type=int, default=500, help="Steps between checkpointing")
    parser.add_argument("--exp_name", type=str, default="marathi_conformer_ssl", help="Experiment name")
    parser.add_argument("--run_dir", type=str, default="run3", help="Directory to save the run")
    parser.add_argument("--stage", type=str, default="pretrain", help="Semantic training stage")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    # Caching arguments
    parser.add_argument("--cache_dir", type=str, default="data_cache/pretrain", help="Directory for local disk cache shards")
    parser.add_argument("--cache_chunk_size", type=int, default=10000, help="Number of samples per local cache shard")
    parser.add_argument("--max_cached_chunks", type=int, default=10, help="Maximum number of cache shards on disk")
    # Early stopping arguments
    parser.add_argument("--early_stop_patience", type=int, default=2, help="Checkpoints without improvement before stopping")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4, help="Minimum improvement delta")
    parser.add_argument("--early_stop_metric", type=str, default="masked_l1", help="Metric to monitor for early stopping")

    args = parser.parse_args()

    train_pretrain(
        run_dir=args.run_dir,
        epochs=args.epochs,
        total_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        plot_every=args.plot_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        exp_name=args.exp_name,
        stage=args.stage,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        cache_dir=args.cache_dir,
        cache_chunk_size=args.cache_chunk_size,
        max_cached_chunks=args.max_cached_chunks,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_metric=args.early_stop_metric,
    )
