"""
Automated Batch Size Finder for Streaming Conformer on NVIDIA RTX A5000 (24GB).
Benchmarks forward + backward pass + optimizer step across increasing batch sizes
to identify the throughput sweet spot and maximum safe VRAM threshold.
"""

import sys
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import StreamingASRModel


def benchmark_batch_size(
    batch_size: int,
    audio_duration_sec: float = 8.0,
    sample_rate: int = 16000,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
    device: torch.device = torch.device("cuda"),
):
    """
    Simulates training forward + backward pass for a given batch size.
    Returns performance metrics or None if CUDA OOM.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    num_samples = int(audio_duration_sec * sample_rate)

    # Initialize model on GPU
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

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    # Generate synthetic batch of realistic speech
    inputs = torch.randn(batch_size, num_samples, device=device)

    try:
        # Warmup iterations
        for _ in range(warmup_iters):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model.forward_pretrain(inputs)
                orig = outputs["original_spec"]
                recon = outputs["reconstructed_spec"]
                mask = outputs["mask"]
                mask_3d = mask.unsqueeze(-1).expand_as(orig)
                loss = torch.abs(orig[mask_3d] - recon[mask_3d]).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()

        # Benchmark iterations
        start_time = time.time()
        for _ in range(benchmark_iters):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model.forward_pretrain(inputs)
                orig = outputs["original_spec"]
                recon = outputs["reconstructed_spec"]
                mask = outputs["mask"]
                mask_3d = mask.unsqueeze(-1).expand_as(orig)
                loss = torch.abs(orig[mask_3d] - recon[mask_3d]).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        avg_step_time_ms = (total_time / benchmark_iters) * 1000.0
        total_audio_sec = batch_size * audio_duration_sec * benchmark_iters
        throughput_audio_sec_per_sec = total_audio_sec / total_time
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        del model, optimizer, inputs, outputs, loss
        torch.cuda.empty_cache()

        return {
            "batch_size": batch_size,
            "step_time_ms": avg_step_time_ms,
            "throughput_aud_sec_per_sec": throughput_audio_sec_per_sec,
            "rtf": total_time / total_audio_sec,
            "peak_vram_gb": peak_vram_gb,
            "oom": False,
        }

    except torch.cuda.OutOfMemoryError:
        del model, optimizer, inputs
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size,
            "step_time_ms": None,
            "throughput_aud_sec_per_sec": 0.0,
            "rtf": None,
            "peak_vram_gb": None,
            "oom": True,
        }


def search_optimal_batch_size():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU.")
        return

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"================================================================================")
    print(f"SEARCHING OPTIMAL BATCH SIZE on {device_name} ({total_vram_gb:.2f} GB VRAM)")
    print(f"Audio Duration: 8.0s clips @ 16kHz | Model: Conformer-12 (20.5M params, FP16 AMP)")
    print(f"================================================================================\n")

    candidate_batch_sizes = [16, 32, 64, 96, 128, 192, 256, 320, 384, 512]
    results = []

    print(f"{'Batch Size':>10s} | {'Step Time':>12s} | {'Throughput':>18s} | {'Training RTF':>12s} | {'Peak VRAM':>12s} | {'Status':>8s}", flush=True)
    print("-" * 84, flush=True)

    for bs in candidate_batch_sizes:
        res = benchmark_batch_size(bs)
        results.append(res)

        if res["oom"]:
            print(f"{bs:10d} | {'OOM':>12s} | {'0.0 aud-s/s':>18s} | {'N/A':>12s} | {'Exceeded 24GB':>12s} | {'FAIL':>8s}", flush=True)
            print(f"\n[!] Out Of Memory encountered at Batch Size = {bs}. Halting further increases.", flush=True)
            break
        else:
            vram_pct = (res['peak_vram_gb'] / total_vram_gb) * 100.0
            print(
                f"{bs:10d} | "
                f"{res['step_time_ms']:10.1f} ms | "
                f"{res['throughput_aud_sec_per_sec']:10.1f} aud-s/s | "
                f"{res['rtf']:10.4f}  | "
                f"{res['peak_vram_gb']:6.2f} GB ({vram_pct:4.1f}%) | "
                f"{'PASS':>8s}",
                flush=True,
            )

    # Determine optimal recommendation
    successful = [r for r in results if not r["oom"]]
    if not successful:
        print("\nAll tested batch sizes failed.")
        return

    # Sort by throughput
    best_throughput = max(successful, key=lambda x: x["throughput_aud_sec_per_sec"])

    # Ideal sweet spot: high throughput with 50-70% VRAM headroom to handle occasional 12-14s audio clips
    safe_candidates = [r for r in successful if r["peak_vram_gb"] <= 0.70 * total_vram_gb]
    best_safe = max(safe_candidates, key=lambda x: x["throughput_aud_sec_per_sec"]) if safe_candidates else best_throughput

    print("\n" + "=" * 84)
    print(f"RECOMMENDATION SUMMARY:")
    print(f"  * Highest Raw Throughput Batch Size : B = {best_throughput['batch_size']} ({best_throughput['throughput_aud_sec_per_sec']:.1f} audio-sec/sec, {best_throughput['peak_vram_gb']:.2f} GB VRAM)")
    print(f"  * Recommended Production Batch Size : B = {best_safe['batch_size']} ({best_safe['throughput_aud_sec_per_sec']:.1f} audio-sec/sec, {best_safe['peak_vram_gb']:.2f} GB VRAM, {best_safe['peak_vram_gb']/total_vram_gb*100:.1f}% VRAM)")
    print(f"    (Leaving {(total_vram_gb - best_safe['peak_vram_gb']):.2f} GB safety headroom for variable-length 12-14s clips)")
    print("=" * 84)


if __name__ == "__main__":
    search_optimal_batch_size()
