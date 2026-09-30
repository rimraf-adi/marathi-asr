"""
Comprehensive Hardware & Efficiency Profiling Suite for Marathi ASR.
Compares:
  1. Our Multi-Dialect MoE Conformer (Sparse Top-2 Routing across 4 Experts)
  2. ARTPARK-IISc / SraVaani 1.0 (Zipformer2 + CTC)
  3. AI4Bharat / Indic Conformer 600M Multilingual (Dense CTC via ONNX)

Measures:
  - Total vs Active Parameter Count & Sparsity Ratio
  - Static Model Size on Disk & Static In-Memory Footprint (MB)
  - Real-Time Factor (RTF), P50, P90, P99 Latency across 3s, 8s, 15s audio lengths
  - Batched GPU Throughput (Hours of audio / minute) & Peak VRAM across B in [1, 4, 8, 16, 32, 64]
  - CPU Scaling (1, 4, 8, 16 threads)
  - Decoding Overhead (Pure Greedy vs Lexicon Beam vs KenLM Default vs KenLM Soft)

Outputs:
  - inference/results/hardware/hardware_benchmark_summary.csv
  - inference/results/hardware/latency_by_duration.csv
  - inference/results/hardware/batch_throughput_scaling.csv
  - inference/results/hardware/cpu_scaling.csv
  - inference/results/hardware/decoder_overhead.csv
  - inference/results/hardware/hardware_report.md
"""

import os
import sys
import gc
import glob
import time
import json
import csv
import platform
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Windows UTF-8 configuration
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(root_dir, ".env"))

import torch
import torchaudio
import soundfile as sf
from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from moe.upcycling import load_moe_model
from decoding.beam_search_decoder import MarathiBeamSearchDecoder, load_marathi_wordlist
from inference.pipeline import SraVaaniModel, IndicConformerModel


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_benchmark_wavs() -> List[str]:
    patterns = [
        "data/vistaar_benchmarks/kathbath_noisy/marathi/wavs/*.wav",
        "data/vistaar_benchmarks/indictts/marathi/wavs/*.wav",
        "data/vistaar_benchmarks/kathbath/marathi/wavs/*.wav",
    ]
    for p in patterns:
        files = glob.glob(p)
        if files:
            return sorted(files)
    raise FileNotFoundError("No benchmark wav files found on disk!")


def run_full_hardware_benchmark(output_dir: str = "inference/results/hardware"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
    wav_files = get_benchmark_wavs()

    print("=" * 80)
    print("COMPREHENSIVE HARDWARE BENCHMARK FOR MARATHI ASR")
    print(f"  Platform:    {platform.platform()}")
    print(f"  CPU Cores:   {os.cpu_count()}")
    print(f"  GPU Device:  {gpu_name} ({total_vram_gb:.2f} GB VRAM, CUDA {torch.version.cuda})")
    print(f"  Test Wavs:   {len(wav_files)} audio samples available on disk")
    print("=" * 80)

    model_profiles = []
    all_latency_records = []
    all_batch_records = []
    all_cpu_records = []

    # =========================================================================
    # 1. OUR MODEL: Multi-Dialect MoE Conformer
    # =========================================================================
    print("\n" + "="*50)
    print("PROFILING [1/3]: Our Multi-Dialect MoE Conformer")
    print("="*50)
    ckpt_path = "runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt"
    ckpt_size_mb = Path(ckpt_path).stat().st_size / (1024 ** 2)

    torch.cuda.empty_cache()
    gc.collect()
    mem_before = get_vram_mb()
    model_our = load_moe_model(ckpt_path, device=torch.device(device))
    model_our.eval()
    mem_after = get_vram_mb()
    static_vram_our = mem_after - mem_before

    total_p, train_p = count_parameters(model_our)
    dense_params = sum(p.numel() for n, p in model_our.named_parameters() if "expert" not in n)
    expert_params = sum(p.numel() for n, p in model_our.named_parameters() if "expert" in n)
    active_p = dense_params + (expert_params // 2)

    model_profiles.append({
        "model_name": "Our Model (MoE Conformer)",
        "architecture": "Conformer + Sparse MoE (Top-2 / 4 Experts)",
        "total_parameters_m": round(total_p / 1e6, 2),
        "active_parameters_m": round(active_p / 1e6, 2),
        "sparsity_ratio": f"{active_p / total_p * 100:.1f}%",
        "checkpoint_disk_mb": round(ckpt_size_mb, 1),
        "static_vram_mb": round(static_vram_our, 1),
    })

    # Latency across 3s, 8s, 15s
    for dur in [3.0, 8.0, 15.0]:
        audio = torch.randn(1, int(dur * 16000), dtype=torch.float32, device=device)
        for _ in range(3):
            with torch.no_grad():
                _ = model_our.forward_ctc(audio, chunk_size=None)
        torch.cuda.synchronize()

        times = []
        reset_peak_vram()
        for _ in range(15):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model_our.forward_ctc(audio, chunk_size=None)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        lat_arr = np.array(times)
        mean_l = float(np.mean(lat_arr))
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p99 = float(np.percentile(lat_arr, 99))
        rtf = (mean_l / 1000.0) / dur
        peak_vram = get_peak_vram_mb()
        print(f"  Duration {dur:4.1f}s | Mean: {mean_l:6.1f} ms | P50: {p50:6.1f} ms | P90: {p90:6.1f} ms | RTF: {rtf:0.4f}x | VRAM: {peak_vram:6.1f} MB")
        all_latency_records.append({
            "model_name": "Our Model (MoE Conformer)",
            "duration_sec": dur,
            "mean_latency_ms": round(mean_l, 2),
            "p50_latency_ms": round(p50, 2),
            "p90_latency_ms": round(p90, 2),
            "p99_latency_ms": round(p99, 2),
            "rtf": round(rtf, 4),
            "peak_vram_mb": round(peak_vram, 2),
        })

    # Batch throughput B in [1, 4, 8, 16, 32, 64]
    dur = 5.0
    for bs in [1, 4, 8, 16, 32, 64]:
        audio = torch.randn(bs, int(dur * 16000), dtype=torch.float32, device=device)
        try:
            for _ in range(2):
                with torch.no_grad():
                    _ = model_our.forward_ctc(audio, chunk_size=None)
            torch.cuda.synchronize()
            reset_peak_vram()
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model_our.forward_ctc(audio, chunk_size=None)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            mean_sec = float(np.mean(times))
            total_audio_sec = bs * dur
            audio_hours_per_min = (total_audio_sec / 3600.0) / (mean_sec / 60.0)
            utts_per_sec = bs / mean_sec
            peak_vram = get_peak_vram_mb()
            print(f"  Batch {bs:2d} | Time: {mean_sec*1000:6.1f} ms | Throughput: {audio_hours_per_min:6.2f} hrs/min ({utts_per_sec:5.1f} utt/s) | VRAM: {peak_vram:6.1f} MB")
            all_batch_records.append({
                "model_name": "Our Model (MoE Conformer)",
                "batch_size": bs,
                "mean_batch_time_ms": round(mean_sec * 1000.0, 2),
                "audio_hours_per_min": round(audio_hours_per_min, 2),
                "utts_per_sec": round(utts_per_sec, 2),
                "rtf": round(mean_sec / total_audio_sec, 4),
                "peak_vram_mb": round(peak_vram, 2),
                "status": "OK"
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {bs:2d} | CUDA Out of Memory!")
            torch.cuda.empty_cache()
            break

    # CPU Scaling
    model_cpu = model_our.to("cpu")
    audio_cpu = torch.randn(1, int(5.0 * 16000), dtype=torch.float32, device="cpu")
    orig_th = torch.get_num_threads()
    for th in [1, 4, 8, 16]:
        if th > os.cpu_count():
            continue
        torch.set_num_threads(th)
        with torch.no_grad():
            _ = model_cpu.forward_ctc(audio_cpu, chunk_size=None)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model_cpu.forward_ctc(audio_cpu, chunk_size=None)
            times.append(time.perf_counter() - t0)
        mean_sec = float(np.mean(times))
        rtf = mean_sec / 5.0
        print(f"  CPU Threads {th:2d} | Latency: {mean_sec*1000:6.1f} ms | RTF: {rtf:0.4f}x")
        all_cpu_records.append({
            "model_name": "Our Model (MoE Conformer)",
            "cpu_threads": th,
            "latency_ms": round(mean_sec * 1000.0, 2),
            "rtf": round(rtf, 4),
        })
    torch.set_num_threads(orig_th)
    del model_cpu

    # Decoder benchmark
    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    words = load_marathi_wordlist(max_words=30000)
    dec_beam = MarathiBeamSearchDecoder(tokenizer, unigram_words=words, kenlm_path=None)
    dec_kenlm_def = MarathiBeamSearchDecoder(tokenizer, unigram_words=words, kenlm_path="lm/marathi_5gram.binary", alpha=0.5, beta=1.5)
    dec_kenlm_soft = MarathiBeamSearchDecoder(tokenizer, unigram_words=words, kenlm_path="lm/marathi_5gram.binary", alpha=0.15, beta=1.0)

    model_our = model_our.to(device)
    with torch.no_grad():
        test_audio = torch.randn(1, 16000 * 5, dtype=torch.float32, device=device)
        logits = model_our.forward_ctc(test_audio)["log_probs"].squeeze(0).cpu().float().numpy()

    decoder_records = []
    print("\n--- Decoder Execution Overhead (5.0s audio) ---")
    dec_list = [
        ("Pure Greedy CTC (argmax)", lambda l: sanitize_transcript(tokenizer.ctc_decode(l.argmax(axis=-1).tolist()))),
        ("Prefix Beam Search (Lexicon only)", lambda l: dec_beam.decode(l, beam_width=16)),
        ("Prefix Beam + KenLM 5-gram Default (α=0.5)", lambda l: dec_kenlm_def.decode(l, beam_width=16)),
        ("Prefix Beam + KenLM 5-gram Soft (α=0.15)", lambda l: dec_kenlm_soft.decode(l, beam_width=16)),
    ]
    for name, fn in dec_list:
        _ = fn(logits)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = fn(logits)
            times.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = float(np.mean(times))
        p50 = float(np.percentile(times, 50))
        p90 = float(np.percentile(times, 90))
        print(f"  {name:45s} | Mean: {mean_ms:6.2f} ms | P50: {p50:6.2f} ms | P90: {p90:6.2f} ms")
        decoder_records.append({
            "decoder_name": name,
            "mean_latency_ms": round(mean_ms, 2),
            "p50_latency_ms": round(p50, 2),
            "p90_latency_ms": round(p90, 2),
        })

    del model_our
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # 2. SraVaani 1.0 (ARTPARK-IISc)
    # =========================================================================
    print("\n" + "="*50)
    print("PROFILING [2/3]: ARTPARK-IISc / SraVaani 1.0")
    print("="*50)
    torch.cuda.empty_cache()
    gc.collect()
    mem_before = get_vram_mb()
    sravaani = SraVaaniModel(device=device)
    mem_after = get_vram_mb()
    static_vram_sra = mem_after - mem_before
    total_p, _ = count_parameters(sravaani.model)

    model_profiles.append({
        "model_name": "SraVaani 1.0 (ARTPARK-IISc)",
        "architecture": "Zipformer2 + CTC",
        "total_parameters_m": round(total_p / 1e6, 2),
        "active_parameters_m": round(total_p / 1e6, 2),
        "sparsity_ratio": "100.0% (Dense)",
        "checkpoint_disk_mb": 158.0,
        "static_vram_mb": round(static_vram_sra, 1),
    })

    # Latency across test audio files
    sample_wav = wav_files[0]
    info = sf.info(sample_wav)
    sample_dur = info.duration

    # Warmup
    _ = sravaani.transcribe_paths([sample_wav], batch_size=1)
    torch.cuda.synchronize()

    times = []
    reset_peak_vram()
    for _ in range(10):
        t0 = time.perf_counter()
        _ = sravaani.transcribe_paths([sample_wav], batch_size=1)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(times)
    mean_l = float(np.mean(lat_arr))
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))
    rtf = (mean_l / 1000.0) / sample_dur
    peak_vram = get_peak_vram_mb()
    print(f"  Single Audio ({sample_dur:4.1f}s) | Mean: {mean_l:6.1f} ms | P50: {p50:6.1f} ms | P90: {p90:6.1f} ms | RTF: {rtf:0.4f}x | VRAM: {peak_vram:6.1f} MB")
    all_latency_records.append({
        "model_name": "SraVaani 1.0 (ARTPARK-IISc)",
        "duration_sec": round(sample_dur, 2),
        "mean_latency_ms": round(mean_l, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p99_latency_ms": round(p99, 2),
        "rtf": round(rtf, 4),
        "peak_vram_mb": round(peak_vram, 2),
    })

    # Batch throughput B in [1, 4, 8, 16, 32]
    for bs in [1, 4, 8, 16, 32]:
        batch_paths = (wav_files * (bs // len(wav_files) + 1))[:bs]
        total_audio_sec = sum(sf.info(p).duration for p in batch_paths)
        try:
            _ = sravaani.transcribe_paths(batch_paths, batch_size=bs)
            torch.cuda.synchronize()
            reset_peak_vram()

            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = sravaani.transcribe_paths(batch_paths, batch_size=bs)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            mean_sec = float(np.mean(times))
            audio_hours_per_min = (total_audio_sec / 3600.0) / (mean_sec / 60.0)
            utts_per_sec = bs / mean_sec
            peak_vram = get_peak_vram_mb()
            print(f"  Batch {bs:2d} | Time: {mean_sec*1000:6.1f} ms | Throughput: {audio_hours_per_min:6.2f} hrs/min ({utts_per_sec:5.1f} utt/s) | VRAM: {peak_vram:6.1f} MB")
            all_batch_records.append({
                "model_name": "SraVaani 1.0 (ARTPARK-IISc)",
                "batch_size": bs,
                "mean_batch_time_ms": round(mean_sec * 1000.0, 2),
                "audio_hours_per_min": round(audio_hours_per_min, 2),
                "utts_per_sec": round(utts_per_sec, 2),
                "rtf": round(mean_sec / total_audio_sec, 4),
                "peak_vram_mb": round(peak_vram, 2),
                "status": "OK"
            })
        except Exception as e:
            print(f"  Batch {bs:2d} | Error: {e}")
            break

    del sravaani
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # 3. Indic Conformer 600M (AI4Bharat)
    # =========================================================================
    print("\n" + "="*50)
    print("PROFILING [3/3]: AI4Bharat / Indic Conformer 600M Multilingual")
    print("="*50)
    torch.cuda.empty_cache()
    gc.collect()
    indic = IndicConformerModel(device="cpu")
    total_p, _ = count_parameters(indic.model)

    model_profiles.append({
        "model_name": "Indic Conformer 600M (AI4Bharat)",
        "architecture": "Conformer Dense (600M Multilingual)",
        "total_parameters_m": round(total_p / 1e6, 2),
        "active_parameters_m": round(total_p / 1e6, 2),
        "sparsity_ratio": "100.0% (Dense)",
        "checkpoint_disk_mb": 2400.0,
        "static_vram_mb": 0.0,  # Runs on ONNX CPU
    })

    # Single-utterance latency
    _ = indic.transcribe_single(sample_wav)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = indic.transcribe_single(sample_wav)
        times.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(times)
    mean_l = float(np.mean(lat_arr))
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))
    rtf = (mean_l / 1000.0) / sample_dur
    print(f"  Single Audio ({sample_dur:4.1f}s) | Mean: {mean_l:6.1f} ms | P50: {p50:6.1f} ms | P90: {p90:6.1f} ms | RTF: {rtf:0.4f}x")
    all_latency_records.append({
        "model_name": "Indic Conformer 600M (AI4Bharat)",
        "duration_sec": round(sample_dur, 2),
        "mean_latency_ms": round(mean_l, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p99_latency_ms": round(p99, 2),
        "rtf": round(rtf, 4),
        "peak_vram_mb": 0.0,
    })

    # CPU thread scaling for Indic Conformer
    orig_th = torch.get_num_threads()
    for th in [1, 4, 8, 16]:
        if th > os.cpu_count():
            continue
        torch.set_num_threads(th)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = indic.transcribe_single(sample_wav)
            times.append(time.perf_counter() - t0)
        mean_sec = float(np.mean(times))
        rtf = mean_sec / sample_dur
        print(f"  CPU Threads {th:2d} | Latency: {mean_sec*1000:6.1f} ms | RTF: {rtf:0.4f}x")
        all_cpu_records.append({
            "model_name": "Indic Conformer 600M (AI4Bharat)",
            "cpu_threads": th,
            "latency_ms": round(mean_sec * 1000.0, 2),
            "rtf": round(rtf, 4),
        })
    torch.set_num_threads(orig_th)

    del indic
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # SAVE CSVs & MARKDOWN REPORT
    # =========================================================================
    sum_csv = out_dir / "hardware_benchmark_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_profiles[0].keys()))
        writer.writeheader()
        writer.writerows(model_profiles)

    lat_csv = out_dir / "latency_by_duration.csv"
    with open(lat_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_latency_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_latency_records)

    batch_csv = out_dir / "batch_throughput_scaling.csv"
    with open(batch_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_batch_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_batch_records)

    cpu_csv = out_dir / "cpu_scaling.csv"
    with open(cpu_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_cpu_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_cpu_records)

    dec_csv = out_dir / "decoder_overhead.csv"
    with open(dec_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(decoder_records[0].keys()))
        writer.writeheader()
        writer.writerows(decoder_records)

    rep_md = out_dir / "hardware_report.md"
    with open(rep_md, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Hardware & Efficiency Benchmark: Marathi ASR Models\n\n")
        f.write(f"- **Benchmarking Environment**: `{platform.platform()}`\n")
        f.write(f"- **CPU**: {os.cpu_count()} Cores\n")
        f.write(f"- **GPU**: `{gpu_name}` ({total_vram_gb:.2f} GB VRAM, CUDA {torch.version.cuda})\n\n")

        f.write("## 1. Model Footprint, Active Parameters & Sparsity Ratio\n\n")
        f.write("| Model Name | Architecture | Total Params | Active Params | Sparsity Ratio | Checkpoint Disk (MB) | Static VRAM (MB) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for m in model_profiles:
            f.write(f"| **{m['model_name']}** | {m['architecture']} | {m['total_parameters_m']}M | **{m['active_parameters_m']}M** | {m['sparsity_ratio']} | {m['checkpoint_disk_mb']} MB | {m['static_vram_mb']} MB |\n")

        f.write("\n\n## 2. Latency & Real-Time Factor (RTF) across Audio Durations (Batch Size = 1)\n\n")
        f.write("| Model Name | Audio Duration | Mean Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | RTF | Peak VRAM (MB) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in all_latency_records:
            f.write(f"| **{r['model_name']}** | {r['duration_sec']}s | {r['mean_latency_ms']} ms | {r['p50_latency_ms']} ms | {r['p90_latency_ms']} ms | {r['p99_latency_ms']} ms | **{r['rtf']}x** | {r['peak_vram_mb']} MB |\n")

        f.write("\n\n## 3. GPU Batch Scaling & Throughput (Hours of Audio / Minute)\n\n")
        f.write("| Model Name | Batch Size | Latency (ms) | Throughput (Audio Hrs / Min) | Utterances / Sec | RTF | Peak VRAM (MB) | Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for b in all_batch_records:
            f.write(f"| **{b['model_name']}** | B={b['batch_size']} | {b['mean_batch_time_ms']} ms | **{b['audio_hours_per_min']} hrs/min** | {b['utts_per_sec']} utt/s | {b['rtf']}x | {b['peak_vram_mb']} MB | {b['status']} |\n")

        f.write("\n\n## 4. CPU Thread Scaling (Edge / Serverless without GPU)\n\n")
        f.write("| Model Name | CPU Threads | Latency (5s Audio) | RTF |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for c in all_cpu_records:
            f.write(f"| **{c['model_name']}** | {c['cpu_threads']} Threads | {c['latency_ms']} ms | **{c['rtf']}x** |\n")

        f.write("\n\n## 5. Decoder Overhead Breakdown (Our Model)\n\n")
        f.write("| Decoding Strategy | Mean Latency (ms) | P50 (ms) | P90 (ms) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for d in decoder_records:
            f.write(f"| **{d['decoder_name']}** | {d['mean_latency_ms']} ms | {d['p50_latency_ms']} ms | {d['p90_latency_ms']} ms |\n")

    print("\n" + "="*80)
    print("ALL HARDWARE BENCHMARKS COMPLETED SUCCESSFULLY!")
    print(f"  Summary CSV:       {sum_csv}")
    print(f"  Latency CSV:       {lat_csv}")
    print(f"  Batch Scaling CSV: {batch_csv}")
    print(f"  CPU Scaling CSV:   {cpu_csv}")
    print(f"  Decoder CSV:       {dec_csv}")
    print(f"  Master Report:     {rep_md}")
    print("="*80)


if __name__ == "__main__":
    run_full_hardware_benchmark()
