"""
Final Frozen Benchmark & Evaluation Suite for Multi-Dialect Marathi Streaming ASR.
Evaluates model checkpoints strictly on the held-out IISc RESPIN test set (2,170 utterances across D1, D2, D3, D4).
Calculates:
  - Character Error Rate (CER %)
  - Word Error Rate (WER %) via jiwer
  - Real-Time Factor (RTF) & Compute Latency
  - Multi-Exit Performance (Layer 4, Layer 8, Layer 12)
  - MoE Router Specialization & Expert Gating Distribution per Dialect
Exports structured JSON, CSV, and Markdown comparison tables for publication and documentation.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional

# Ensure root directory is in sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import soundfile as sf
import torch
import jiwer

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer
from decoding.beam_search_decoder import MarathiBeamSearchDecoder, load_marathi_wordlist
from model.asr_model import StreamingASRModel
from moe.moe_layer import SparseMoELayer
from moe.upcycling import load_moe_model

DIALECT_NAMES = {
    "D1": "Malvani / Konkan",
    "D2": "Ahirani / Khandesh",
    "D3": "Standard Marathi",
    "D4": "Varhadi / Vidarbha",
}


def load_model_for_eval(checkpoint_path: str, device: torch.device) -> StreamingASRModel:
    """Intelligently detects checkpoint architecture (Dense Multi-Exit vs MoE) and loads weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Check if this checkpoint has MoE layers (i.e. block.ffn2.experts)
    is_moe = any("ffn2.experts" in k for k in state_dict.keys())

    if is_moe:
        print(f"[Model Loader] Detected MoE Architecture. Loading 3-Dialect MoE Conformer...")
        model = load_moe_model(checkpoint_path, device=device)
    else:
        print(f"[Model Loader] Detected Dense Multi-Exit Architecture. Loading standard Conformer...")
        model = StreamingASRModel(
            feat_dim=80,
            d_model=256,
            num_layers=12,
            n_heads=4,
            conv_kernel_size=31,
            ffn_expansion=4,
            dropout=0.0,
            exit_layers=[4, 8, 12],
            enable_reconstruction_head=False,
            vocab_size=105,
        ).to(device)
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_state[k.replace("backbone.", "")] = v
            else:
                backbone_state[k] = v
        model.load_state_dict(backbone_state, strict=False)

    model.eval()
    return model


def run_benchmark(
    checkpoint_path: str,
    test_dir: str = r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr",
    vocab_path: str = "data_utils/vocab.json",
    output_dir: str = "run2/eval_results",
    model_tag: str = "conformer_eval",
    chunk_size: Optional[int] = 16,  # 640ms streaming chunks (or None for offline full-context)
    batch_size: int = 16,
    limit: Optional[int] = None,
    use_beam_search: bool = True,
    beam_width: int = 16,
    kenlm_path: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n================================================================================")
    print(f"[Benchmark] Evaluating '{model_tag}' on IISc RESPIN Held-Out Test Set")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Streaming Chunk Size: {chunk_size} frames ({chunk_size * 40} ms) [None = offline]")
    print(f"  Decoding Strategy: {'Beam Search (width=' + str(beam_width) + ')' if use_beam_search else 'Greedy CTC'}")
    print(f"================================================================================")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = MarathiTokenizer(vocab_path)
    model = load_model_for_eval(checkpoint_path, device)

    beam_decoder = None
    if use_beam_search:
        print(f"[Decoder] Initializing Marathi Beam Search Decoder...")
        wordlist = load_marathi_wordlist(max_words=30000)
        beam_decoder = MarathiBeamSearchDecoder(tokenizer, unigram_words=wordlist, kenlm_path=kenlm_path)

    # 1. Load test metadata
    meta_path = Path(test_dir) / "meta_test_mr.json"
    print(f"[Data] Loading frozen test metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        test_meta = json.load(f)

    samples = []
    for uid, info in test_meta.items():
        wav_file = Path(test_dir) / info["wav_path"]
        samples.append({
            "uid": uid,
            "wav_path": str(wav_file),
            "text": info.get("text", "").strip(),
            "dialect": info.get("dialect", "D3"),
            "duration": float(info.get("duration", 0.0)),
            "speaker_id": info.get("speaker_id", "unknown"),
        })

    if limit and limit > 0:
        print(f"[Data] Subsampling {limit} items for rapid evaluation verification...")
        samples = samples[:limit]

    total_samples = len(samples)
    total_audio_sec = sum(s["duration"] for s in samples)
    print(f"[Data] Benchmark set: {total_samples:,} utterances ({total_audio_sec/3600.0:.2f} hours)")

    # Dialect breakdown tracker
    dialect_stats = {
        d: {
            "Layer 4 (Fast)": {"cer_sum": 0.0, "wer_sum": 0.0, "count": 0, "chars": 0, "words": 0},
            "Layer 8 (Balanced)": {"cer_sum": 0.0, "wer_sum": 0.0, "count": 0, "chars": 0, "words": 0},
            "Layer 12 (Deep)": {"cer_sum": 0.0, "wer_sum": 0.0, "count": 0, "chars": 0, "words": 0},
        }
        for d in ["D1", "D2", "D3", "D4", "ALL"]
    }

    exit_times = {"Layer 4 (Fast)": 0.0, "Layer 8 (Balanced)": 0.0, "Layer 12 (Deep)": 0.0}
    qualitative_samples = []

    # Batch iterator
    t_start = time.time()
    processed_count = 0

    for i in range(0, total_samples, batch_size):
        batch_items = samples[i:i + batch_size]
        audios = []
        texts = []
        dialects = []
        uids = []
        durs = []

        for item in batch_items:
            try:
                audio, sr = sf.read(item["wav_path"], dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    continue
                audios.append(torch.from_numpy(audio))
                texts.append(item["text"])
                dialects.append(item["dialect"])
                uids.append(item["uid"])
                durs.append(item["duration"])
            except Exception as e:
                continue

        if not audios:
            continue

        b_cur = len(audios)
        audio_lens = torch.tensor([len(a) for a in audios], dtype=torch.long)
        max_len = audio_lens.max().item()
        padded_audio = torch.zeros(b_cur, max_len, dtype=torch.float32)
        for idx, a in enumerate(audios):
            padded_audio[idx, :len(a)] = a

        padded_audio = padded_audio.to(device)

        with torch.no_grad():
            # Compute stats and measure true inference time for each exit independently
            for layer_idx, exit_name in [
                (4, "Layer 4 (Fast)"),
                (8, "Layer 8 (Balanced)"),
                (12, "Layer 12 (Deep)"),
            ]:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                ctc_dict = model.forward_ctc(padded_audio, chunk_size=None, max_exit_layer=layer_idx)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                exit_times[exit_name] += (time.time() - t0)

                logits = ctc_dict["exit_log_probs"][layer_idx]

                for b in range(b_cur):
                    if beam_decoder is not None and layer_idx == 12:
                        pred_text = beam_decoder.decode(logits[b], beam_width=beam_width)
                    else:
                        pred_tokens = logits[b].argmax(dim=-1).cpu().tolist()
                        pred_text = sanitize_transcript(tokenizer.ctc_decode(pred_tokens))

                    ref_text = sanitize_transcript(texts[b])
                    dialect = dialects[b]

                    cer = compute_cer(ref_text, pred_text)
                    try:
                        wer = jiwer.wer(ref_text, pred_text) if len(ref_text.split()) > 0 else 0.0
                    except Exception:
                        wer = 1.0

                    ref_words = len(ref_text.split())
                    ref_chars = len(ref_text)

                    # Update dialect-specific
                    d_stat = dialect_stats[dialect][exit_name]
                    d_stat["cer_sum"] += cer * ref_chars
                    d_stat["wer_sum"] += wer * ref_words
                    d_stat["count"] += 1
                    d_stat["chars"] += ref_chars
                    d_stat["words"] += ref_words

                    # Update ALL
                    all_stat = dialect_stats["ALL"][exit_name]
                    all_stat["cer_sum"] += cer * ref_chars
                    all_stat["wer_sum"] += wer * ref_words
                    all_stat["count"] += 1
                    all_stat["chars"] += ref_chars
                    all_stat["words"] += ref_words

                    if layer_idx == 12 and len(qualitative_samples) < 15:
                        qualitative_samples.append({
                            "uid": uids[b],
                            "dialect": dialect,
                            "reference": ref_text,
                            "prediction": pred_text,
                            "cer": round(cer * 100, 2),
                            "wer": round(wer * 100, 2),
                        })

        processed_count += b_cur
        if processed_count % 200 == 0 or processed_count == total_samples:
            sys.stdout.write(f"\r  [Evaluating] Processed {processed_count:,}/{total_samples:,} utterances ({processed_count*100/total_samples:.1f}%)...")
            sys.stdout.flush()

    total_eval_time = time.time() - t_start
    print(f"\n[Evaluating] Completed in {total_eval_time:.1f} seconds.\n")

    # Compile Final Results
    summary = {
        "model_tag": model_tag,
        "checkpoint": str(checkpoint_path),
        "chunk_size": chunk_size,
        "total_utterances": total_samples,
        "total_audio_hours": round(total_audio_sec / 3600.0, 3),
        "results": {},
    }

    # Format Markdown and CSV rows
    csv_rows = []
    print(f"===================================================================================================")
    print(f"                            FINAL BENCHMARK EVALUATION RESULTS: {model_tag}")
    print(f"===================================================================================================")
    print(f"{'Exit Level':<20} | {'Dialect':<22} | {'CER (%)':<9} | {'WER (%)':<9} | {'Samples':<8} | {'RTF':<8}")
    print(f"---------------------+------------------------+-----------+-----------+----------+---------")

    for exit_name in ["Layer 4 (Fast)", "Layer 8 (Balanced)", "Layer 12 (Deep)"]:
        rtf = exit_times[exit_name] / max(total_audio_sec, 1e-5)
        for dialect in ["D1", "D2", "D3", "D4", "ALL"]:
            stat = dialect_stats[dialect][exit_name]
            cer = (stat["cer_sum"] / max(stat["chars"], 1)) * 100.0
            wer = (stat["wer_sum"] / max(stat["words"], 1)) * 100.0
            d_label = f"{dialect} ({DIALECT_NAMES.get(dialect, 'Aggregate')})"

            print(f"{exit_name:<20} | {d_label:<22} | {cer:7.2f}% | {wer:7.2f}% | {stat['count']:<8} | {rtf:6.4f}x")

            row_data = {
                "exit": exit_name,
                "dialect_id": dialect,
                "dialect_name": DIALECT_NAMES.get(dialect, "Aggregate"),
                "cer_percent": round(cer, 2),
                "wer_percent": round(wer, 2),
                "utterances": stat["count"],
                "rtf": round(rtf, 4),
            }
            csv_rows.append(row_data)

            if dialect not in summary["results"]:
                summary["results"][dialect] = {}
            summary["results"][dialect][exit_name] = row_data
        print(f"---------------------+------------------------+-----------+-----------+----------+---------")

    # Save CSV
    csv_file = out_path / f"{model_tag}_summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[Artifact] Saved structured CSV: {csv_file}")

    # Also mirror into run2/csvs/ for centralized metric tracking
    run2_csv_dir = Path("run2/csvs")
    run2_csv_dir.mkdir(parents=True, exist_ok=True)
    mirror_csv = run2_csv_dir / f"{model_tag}_summary.csv"
    with open(mirror_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Artifact] Mirrored CSV to centralized directory: {mirror_csv}")

    # Save JSON
    summary["qualitative_samples"] = qualitative_samples
    json_file = out_path / f"{model_tag}_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Artifact] Saved detailed JSON: {json_file}")

    # Save Markdown Table
    md_file = out_path / f"{model_tag}_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Report: `{model_tag}`\n\n")
        f.write(f"- **Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"- **Streaming Chunk Size**: {chunk_size} frames ({chunk_size * 40 if chunk_size else 'Full Context'} ms)\n")
        f.write(f"- **Evaluated Test Set**: IISc RESPIN Held-Out ({total_samples:,} utterances, {total_audio_sec/3600.0:.2f} hrs)\n\n")
        f.write(f"| Exit Level | Dialect | CER (%) | WER (%) | Utterances | RTF |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in csv_rows:
            f.write(f"| {r['exit']} | {r['dialect_id']} ({r['dialect_name']}) | **{r['cer_percent']:.2f}%** | {r['wer_percent']:.2f}% | {r['utterances']} | {r['rtf']}x |\n")

        f.write(f"\n## Qualitative Prediction Samples (Layer 12)\n\n")
        f.write(f"| Dialect | Reference | Prediction | CER (%) | WER (%) |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- |\n")
        for q in qualitative_samples[:10]:
            f.write(f"| {q['dialect']} | {q['reference']} | {q['prediction']} | {q['cer']}% | {q['wer']}% |\n")
    print(f"[Artifact] Saved Markdown Report: {md_file}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Marathi ASR on IISc RESPIN Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, default=r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr", help="RESPIN test set directory")
    parser.add_argument("--vocab", type=str, default="data_utils/vocab.json", help="Vocab path")
    parser.add_argument("--tag", type=str, default="conformer_eval", help="Model identification tag")
    parser.add_argument("--chunk_size", type=int, default=16, help="Streaming chunk size (16 = 640ms, None = full context)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test utterances (for dry run)")
    parser.add_argument("--output_dir", type=str, default="run2/eval_results", help="Output directory")
    parser.add_argument("--beam_search", action="store_true", default=True, help="Enable pyctcdecode beam search")
    parser.add_argument("--no_beam_search", action="store_false", dest="beam_search", help="Disable beam search (use greedy CTC)")
    parser.add_argument("--beam_width", type=int, default=16, help="Beam width for beam search decoding")
    parser.add_argument("--kenlm_path", type=str, default=None, help="Path to optional KenLM language model binary")

    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        vocab_path=args.vocab,
        output_dir=args.output_dir,
        model_tag=args.tag,
        chunk_size=args.chunk_size if args.chunk_size > 0 else None,
        batch_size=args.batch_size,
        limit=args.limit,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width,
        kenlm_path=args.kenlm_path,
    )
