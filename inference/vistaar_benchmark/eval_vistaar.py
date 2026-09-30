"""
Vistaar Multi-Benchmark Evaluation Suite for Marathi ASR.
Evaluates ASR models across Vistaar Marathi benchmarks:
  - IndicTTS
  - Kathbath
  - Kathbath Hard
  - CommonVoice
  - FLEURS
  - MUCS

Evaluates:
  1. Our Multi-Dialect MoE Conformer (CTC Decoding)
  2. ARTPARK-IISc / SraVaani 1.0 (CTC Decoding)
  3. AI4Bharat / IndicConformer 600M Multilingual (CTC Decoding)
  4. AI4Bharat / IndicWhisper Marathi (Optional)

Outputs structured CSVs and comprehensive Markdown reports under inference/results/vistaar/.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(root_dir, ".env"))

import soundfile as sf
import torch
import torchaudio
import jiwer
from tqdm import tqdm

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer
from moe.upcycling import load_moe_model
from inference.pipeline import normalize_marathi_text, OurASRModel, SraVaaniModel, IndicConformerModel


def load_vistaar_manifest(manifest_path: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads a JSONL manifest in Vistaar format and resolves relative audio paths."""
    samples = []
    manifest_p = Path(manifest_path).resolve()
    base_search_dirs = [
        manifest_p.parent,
        manifest_p.parent.parent,
        manifest_p.parent.parent.parent,
        Path("data/vistaar_benchmarks").resolve(),
    ]
    if base_dir:
        base_search_dirs.insert(0, Path(base_dir).resolve())

    with open(manifest_p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            raw_audio = data.get("audio_filepath") or data.get("audio_path")
            audio_path = Path(raw_audio)

            if not audio_path.exists():
                for b in base_search_dirs:
                    cand = b / raw_audio
                    if cand.exists():
                        audio_path = cand
                        break

            text = data.get("text", "").strip()
            duration = float(data.get("duration", 0.0))
            if not duration and audio_path.exists():
                try:
                    info = sf.info(str(audio_path))
                    duration = info.duration
                except Exception:
                    duration = 0.0

            samples.append({
                "uid": f"sample_{idx:05d}",
                "wav_path": str(audio_path),
                "text": text,
                "duration": duration,
            })
    return samples


class VistaarEvaluator:
    """Evaluates multiple ASR models on a given Vistaar benchmark."""

    def __init__(
        self,
        output_dir: str = "inference/results/vistaar",
        checkpoint_path: str = "runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt",
        vocab_path: str = "data_utils/vocab.json",
        device: str = "cuda",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.device = device

    def evaluate_model(
        self,
        model_name: str,
        benchmark_name: str,
        samples: List[Dict[str, Any]],
        predictions: List[str],
        compute_time_sec: float,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculates CER, WER, SER, RTF, and detailed token stats."""
        total_audio_sec = sum(s["duration"] for s in samples)
        rtf = compute_time_sec / max(total_audio_sec, 1e-5)
        avg_latency = (compute_time_sec / max(len(samples), 1)) * 1000.0

        raw_cer_sum = 0.0
        raw_wer_sum = 0.0
        norm_cer_sum = 0.0
        norm_wer_sum = 0.0
        chars = 0
        words = 0
        exact_matches = 0
        subs, dels, inss, hits = 0, 0, 0, 0

        detailed_rows = []
        for s, pred in zip(samples, predictions):
            ref = s["text"]
            norm_ref = normalize_marathi_text(ref)
            norm_pred = normalize_marathi_text(pred)

            raw_cer = compute_cer(ref, pred)
            try:
                raw_wer = jiwer.wer(ref, pred) if len(ref.split()) > 0 else 0.0
            except Exception:
                raw_wer = 1.0

            norm_cer = compute_cer(norm_ref, norm_pred)
            measures = jiwer.process_words(norm_ref, norm_pred)
            norm_wer = measures.wer
            is_exact = 1 if norm_ref == norm_pred else 0

            ref_words = len(norm_ref.split())
            ref_chars = len(norm_ref)

            norm_cer_sum += norm_cer * ref_chars
            norm_wer_sum += norm_wer * ref_words
            chars += ref_chars
            words += ref_words
            exact_matches += is_exact
            subs += measures.substitutions
            dels += measures.deletions
            inss += measures.insertions
            hits += measures.hits

            detailed_rows.append({
                "uid": s["uid"],
                "wav_path": s["wav_path"],
                "reference": ref,
                "prediction": pred,
                "normalized_ref": norm_ref,
                "normalized_pred": norm_pred,
                "norm_cer_percent": round(norm_cer * 100, 2),
                "norm_wer_percent": round(norm_wer * 100, 2),
                "exact_match": is_exact,
                "duration_sec": round(s["duration"], 2),
            })

        cer = (norm_cer_sum / max(chars, 1)) * 100.0
        wer = (norm_wer_sum / max(words, 1)) * 100.0
        ser = ((len(samples) - exact_matches) / max(len(samples), 1)) * 100.0
        exact_acc = (exact_matches / max(len(samples), 1)) * 100.0

        summary = {
            "model_name": model_name,
            "benchmark": benchmark_name,
            "cer_percent": round(cer, 2),
            "wer_percent": round(wer, 2),
            "ser_percent": round(ser, 2),
            "exact_match_acc": round(exact_acc, 2),
            "utterances": len(samples),
            "audio_hours": round(total_audio_sec / 3600.0, 3),
            "compute_time_sec": round(compute_time_sec, 2),
            "rtf": round(rtf, 4),
            "latency_ms_per_utt": round(avg_latency, 2),
            "substitutions": subs,
            "deletions": dels,
            "insertions": inss,
            "hits": hits,
        }
        return summary, detailed_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Marathi ASR models on Vistaar benchmarks")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g. indictts, fleurs, kathbath)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL file")
    parser.add_argument("--model", type=str, default="all", choices=["all", "our_model", "sravaani", "indic_conformer"])
    parser.add_argument("--output_dir", type=str, default="inference/results/vistaar")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    samples = load_vistaar_manifest(args.manifest)
    if args.limit:
        samples = samples[:args.limit]

    print(f"\n[Vistaar Evaluation] Benchmark: {args.benchmark.upper()}")
    print(f"Total Utterances: {len(samples)} ({sum(s['duration'] for s in samples)/3600.0:.2f} hours)")

    evaluator = VistaarEvaluator(output_dir=args.output_dir, device=args.device)
    summaries = []

    # 1. Our Model
    if args.model in ["all", "our_model"]:
        print(f"\n--- Running Our Model on {args.benchmark} ---")
        our_model = OurASRModel(checkpoint_path=evaluator.checkpoint_path, vocab_path=evaluator.vocab_path, device=args.device)
        t0 = time.time()
        preds = []
        for i in tqdm(range(0, len(samples), 16), desc="Our Model"):
            batch = samples[i:i + 16]
            audios = []
            for item in batch:
                a, sr = sf.read(item["wav_path"], dtype="float32")
                if len(a.shape) > 1: a = a.mean(axis=1)
                audios.append(torch.from_numpy(a))
            preds.extend(our_model.transcribe_batch(audios, use_beam=False))
        comp_time = time.time() - t0
        s, det = evaluator.evaluate_model("Our Model (MoE Conformer)", args.benchmark, samples, preds, comp_time)
        summaries.append(s)
        det_csv = Path(args.output_dir) / f"{args.benchmark}_our_model_detailed.csv"
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            if det:
                writer = csv.DictWriter(f, fieldnames=list(det[0].keys()))
                writer.writeheader()
                writer.writerows(det)

    # 2. SraVaani
    if args.model in ["all", "sravaani"]:
        print(f"\n--- Running SraVaani on {args.benchmark} ---")
        sravaani = SraVaaniModel(device=args.device)
        t0 = time.time()
        wav_paths = [s["wav_path"] for s in samples]
        preds = []
        for i in tqdm(range(0, len(wav_paths), args.batch_size), desc="SraVaani"):
            batch_paths = wav_paths[i:i + args.batch_size]
            preds.extend(sravaani.transcribe_paths(batch_paths, batch_size=args.batch_size))
        comp_time = time.time() - t0
        s, det = evaluator.evaluate_model("SraVaani 1.0 (ARTPARK-IISc)", args.benchmark, samples, preds, comp_time)
        summaries.append(s)
        det_csv = Path(args.output_dir) / f"{args.benchmark}_sravaani_detailed.csv"
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            if det:
                writer = csv.DictWriter(f, fieldnames=list(det[0].keys()))
                writer.writeheader()
                writer.writerows(det)

    # 3. Indic Conformer
    if args.model in ["all", "indic_conformer"]:
        print(f"\n--- Running Indic Conformer on {args.benchmark} ---")
        indic = IndicConformerModel(device=args.device)
        from concurrent.futures import ThreadPoolExecutor
        def _transcribe(s):
            try: return indic.transcribe_single(s["wav_path"])
            except Exception: return ""
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            preds = list(tqdm(ex.map(_transcribe, samples), total=len(samples), desc="Indic Conformer"))
        comp_time = time.time() - t0
        s, det = evaluator.evaluate_model("Indic Conformer 600M (AI4Bharat)", args.benchmark, samples, preds, comp_time)
        summaries.append(s)

        det_csv = Path(args.output_dir) / f"{args.benchmark}_indic_conformer_detailed.csv"
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            if det:
                writer = csv.DictWriter(f, fieldnames=list(det[0].keys()))
                writer.writeheader()
                writer.writerows(det)

    if summaries:
        sum_csv = Path(args.output_dir) / f"{args.benchmark}_summary.csv"
        with open(sum_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\n[Artifact] Saved benchmark summary: {sum_csv}")

        # Save Markdown Report
        md_file = Path(args.output_dir) / f"{args.benchmark}_report.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Vistaar Benchmark Report: `{args.benchmark.upper()}` (Marathi)\n\n")
            f.write(f"- **Total Utterances**: {len(samples)} ({sum(s['duration'] for s in samples)/3600.0:.2f} hours)\n")
            f.write(f"- **Manifest**: `{args.manifest}`\n\n")
            f.write("| Model Name | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for s in summaries:
                f.write(f"| **{s['model_name']}** | **{s['cer_percent']:.2f}%** | **{s['wer_percent']:.2f}%** | {s['ser_percent']:.2f}% | {s['exact_match_acc']:.2f}% | {s['rtf']}x | {s['latency_ms_per_utt']:.1f} ms |\n")
        print(f"[Artifact] Saved Markdown report: {md_file}")


if __name__ == "__main__":
    main()
