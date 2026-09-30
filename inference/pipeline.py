"""
Unified Marathi ASR Inference & Benchmark Pipeline Suite.
Supports:
  1. Our Multi-Dialect MoE Conformer (Stage 3 Best Model) with CTC Decoding
  2. ARTPARK-IISc / SraVaani 1.0 (FastConformer CTC-TDT Decoder)
  3. AI4Bharat / IndicConformer 600M Multilingual (CTC Decoder)

Evaluates on the frozen IISc RESPIN Marathi Test Set (2,170 utterances across D1, D2, D3, D4)
Outputs structured summary CSVs, detailed per-utterance CSVs, and comprehensive Markdown tables
directly into inference/results/respin/.
"""

import os
import sys
import time
import json
import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(root_dir, ".env"))

import soundfile as sf
import numpy as np
import torch
import torchaudio
import jiwer
from tqdm import tqdm

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer
from moe.upcycling import load_moe_model
from decoding.beam_search_decoder import MarathiBeamSearchDecoder, load_marathi_wordlist

DIALECT_NAMES = {
    "D1": "Malvani / Konkan",
    "D2": "Ahirani / Khandesh",
    "D3": "Standard Marathi",
    "D4": "Varhadi / Vidarbha",
}

PUNCTUATION_CHARS = ["?", "!", ",", ".", ":", ";", '"', "'", "‘", "’", "“", "”", "—", "-", "।", "(", ")", "[", "]", "{", "}"]


def normalize_marathi_text(text: str) -> str:
    """Standardizes Marathi text by stripping annotation tags, punctuation, and extra whitespace."""
    if not text:
        return ""
    text = sanitize_transcript(text)
    for p in PUNCTUATION_CHARS:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()


# ==============================================================================
# 1. Model Wrappers
# ==============================================================================

class OurASRModel:
    """Wrapper for Our Multi-Dialect MoE Conformer ASR Model (CTC Decoding)."""

    def __init__(self, checkpoint_path: str, vocab_path: str = "data_utils/vocab.json", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"[OurModel] Loading checkpoint: {checkpoint_path} onto {self.device}...")
        self.model = load_moe_model(checkpoint_path, device=self.device)
        self.model.eval()
        self.tokenizer = MarathiTokenizer(vocab_path)
        self.beam_decoder = None

    def enable_beam_search(self, beam_width: int = 16, kenlm_path: Optional[str] = None):
        print(f"[OurModel] Initializing Marathi Beam Search Decoder (width={beam_width})...")
        wordlist = load_marathi_wordlist(max_words=30000)
        self.beam_decoder = MarathiBeamSearchDecoder(self.tokenizer, unigram_words=wordlist, kenlm_path=kenlm_path)

    @torch.no_grad()
    def transcribe_batch(self, audio_list: List[torch.Tensor], use_beam: bool = False, beam_width: int = 16) -> List[str]:
        b_cur = len(audio_list)
        lens = torch.tensor([len(a) for a in audio_list], dtype=torch.long)
        max_len = lens.max().item()
        padded = torch.zeros(b_cur, max_len, dtype=torch.float32, device=self.device)
        for idx, a in enumerate(audio_list):
            padded[idx, :len(a)] = a.to(self.device)

        ctc_dict = self.model.forward_ctc(padded, chunk_size=None)
        logits = ctc_dict["log_probs"]

        preds = []
        for b in range(b_cur):
            if use_beam and self.beam_decoder is not None:
                text = self.beam_decoder.decode(logits[b], beam_width=beam_width)
            else:
                tokens = logits[b].argmax(dim=-1).cpu().tolist()
                text = sanitize_transcript(self.tokenizer.ctc_decode(tokens))
            preds.append(text)
        return preds


class SraVaaniModel:
    """Wrapper for ARTPARK-IISc / SraVaani-1.0 ASR Model (CTC Decoding)."""

    def __init__(self, repo_id: str = "ARTPARK-IISc/SraVaani-1.0", token: Optional[str] = None, device: str = "cuda"):
        from transformers import AutoModel
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        token = token or os.environ.get("HF_TOKEN")
        print(f"[SraVaani] Loading model '{repo_id}' onto {self.device}...")
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=token).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def transcribe_paths(self, wav_paths: List[str], batch_size: int = 32) -> List[str]:
        hyps = self.model.transcribe(wav_paths, batch_size=batch_size, return_hypotheses=True)
        preds = []
        for h in hyps:
            raw_text = h.text if hasattr(h, "text") else str(h)
            preds.append(sanitize_transcript(raw_text))
        return preds


class IndicConformerModel:
    """Wrapper for AI4Bharat / IndicConformer 600M Multilingual (CTC Decoding)."""

    def __init__(self, repo_id: str = "ai4bharat/indic-conformer-600m-multilingual", token: Optional[str] = None, device: str = "cuda"):
        from transformers import AutoModel
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        token = token or os.environ.get("HF_TOKEN")
        print(f"[IndicConformer] Loading model '{repo_id}'...")
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=token)
        self.model.eval()

    @torch.no_grad()
    def transcribe_single(self, wav_path: str) -> str:
        wav, sr = torchaudio.load(wav_path)
        if wav.ndim > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
        pred = self.model(wav, "mr", "ctc")
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return sanitize_transcript(str(pred))


# ==============================================================================
# 2. Benchmark Runner
# ==============================================================================

class UnifiedASRPipeline:
    """Complete Pipeline for Marathi ASR Evaluation across Our Model, SraVaani, and IndicConformer."""

    def __init__(
        self,
        test_dir: str = r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr",
        meta_file: str = "meta_test_mr.json",
        output_dir: str = "inference/results/respin",
        checkpoint_path: str = "runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt",
        vocab_path: str = "data_utils/vocab.json",
        device: str = "cuda",
    ):
        self.test_dir = Path(test_dir)
        self.meta_file = self.test_dir / meta_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.device = device
        self.hf_token = os.environ.get("HF_TOKEN")

        # Load metadata
        print(f"\n[Pipeline] Loading IISc RESPIN test metadata from: {self.meta_file}")
        with open(self.meta_file, "r", encoding="utf-8") as f:
            self.test_meta = json.load(f)

        self.samples = []
        for uid, info in self.test_meta.items():
            wav_file = self.test_dir / info["wav_path"]
            self.samples.append({
                "uid": uid,
                "wav_path": str(wav_file),
                "text": info.get("text", "").strip(),
                "dialect": info.get("dialect", "D3"),
                "duration": float(info.get("duration", 0.0)),
                "speaker_id": info.get("speaker_id", "unknown"),
                "domain": info.get("domain", "General"),
            })
        print(f"[Pipeline] Loaded {len(self.samples):,} utterances ({sum(s['duration'] for s in self.samples)/3600.0:.2f} hours)")

    def _evaluate_predictions(self, model_key: str, model_display_name: str, predictions: List[str], samples: List[Dict[str, Any]], compute_time_sec: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculates strict Character Error Rate (CER), Word Error Rate (WER), Sentence Error Rate (SER), and per-dialect stats."""
        total_audio_sec = sum(s["duration"] for s in samples)
        rtf = compute_time_sec / max(total_audio_sec, 1e-5)
        avg_latency_ms = (compute_time_sec / max(len(samples), 1)) * 1000.0

        dialect_stats = {
            d: {
                "raw_cer_sum": 0.0, "raw_wer_sum": 0.0,
                "norm_cer_sum": 0.0, "norm_wer_sum": 0.0,
                "count": 0, "chars": 0, "words": 0,
                "exact_matches": 0, "substitutions": 0,
                "deletions": 0, "insertions": 0, "hits": 0,
                "duration_sec": 0.0,
            }
            for d in ["D1", "D2", "D3", "D4", "ALL"]
        }

        detailed_rows = []
        qualitative_samples = []

        for sample, pred_raw in zip(samples, predictions):
            ref_raw = sample["text"]
            dialect = sample["dialect"]

            # Normalized representations for clean phonemic comparison
            ref_norm = normalize_marathi_text(ref_raw)
            pred_norm = normalize_marathi_text(pred_raw)

            # Raw metrics
            raw_cer = compute_cer(ref_raw, pred_raw)
            try:
                raw_wer = jiwer.wer(ref_raw, pred_raw) if len(ref_raw.split()) > 0 else 0.0
            except Exception:
                raw_wer = 1.0

            # Normalized metrics
            norm_cer = compute_cer(ref_norm, pred_norm)
            norm_measures = jiwer.process_words(ref_norm, pred_norm)
            norm_wer = norm_measures.wer
            is_exact = 1 if ref_norm == pred_norm else 0

            ref_words = len(ref_norm.split())
            ref_chars = len(ref_norm)

            for target_group in [dialect, "ALL"]:
                st = dialect_stats[target_group]
                st["raw_cer_sum"] += raw_cer * len(ref_raw)
                st["raw_wer_sum"] += raw_wer * len(ref_raw.split())
                st["norm_cer_sum"] += norm_cer * ref_chars
                st["norm_wer_sum"] += norm_wer * ref_words
                st["chars"] += ref_chars
                st["words"] += ref_words
                st["count"] += 1
                st["exact_matches"] += is_exact
                st["substitutions"] += norm_measures.substitutions
                st["deletions"] += norm_measures.deletions
                st["insertions"] += norm_measures.insertions
                st["hits"] += norm_measures.hits
                st["duration_sec"] += sample["duration"]

            detailed_rows.append({
                "uid": sample["uid"],
                "dialect": dialect,
                "domain": sample["domain"],
                "duration_sec": sample["duration"],
                "reference": ref_raw,
                "prediction": pred_raw,
                "normalized_ref": ref_norm,
                "normalized_pred": pred_norm,
                "norm_cer_percent": round(norm_cer * 100, 2),
                "norm_wer_percent": round(norm_wer * 100, 2),
                "raw_cer_percent": round(raw_cer * 100, 2),
                "raw_wer_percent": round(raw_wer * 100, 2),
                "exact_match": is_exact,
            })

            if len(qualitative_samples) < 20:
                qualitative_samples.append({
                    "uid": sample["uid"],
                    "dialect": dialect,
                    "reference": ref_raw,
                    "prediction": pred_raw,
                    "cer": round(norm_cer * 100, 2),
                    "wer": round(norm_wer * 100, 2),
                })

        summary_rows = []
        for d in ["D1", "D2", "D3", "D4", "ALL"]:
            st = dialect_stats[d]
            norm_cer = (st["norm_cer_sum"] / max(st["chars"], 1)) * 100.0
            norm_wer = (st["norm_wer_sum"] / max(st["words"], 1)) * 100.0
            ser = ((st["count"] - st["exact_matches"]) / max(st["count"], 1)) * 100.0
            exact_acc = (st["exact_matches"] / max(st["count"], 1)) * 100.0

            summary_rows.append({
                "model_key": model_key,
                "model_name": model_display_name,
                "dialect_id": d,
                "dialect_name": DIALECT_NAMES.get(d, "Aggregate"),
                "cer_percent": round(norm_cer, 2),
                "wer_percent": round(norm_wer, 2),
                "ser_percent": round(ser, 2),
                "exact_match_acc": round(exact_acc, 2),
                "utterances": st["count"],
                "duration_hours": round(st["duration_sec"] / 3600.0, 3),
                "substitutions": st["substitutions"],
                "deletions": st["deletions"],
                "insertions": st["insertions"],
                "hits": st["hits"],
                "rtf": round(rtf, 4),
                "latency_ms_per_utt": round(avg_latency_ms, 2),
            })

        summary_data = {
            "model_key": model_key,
            "model_name": model_display_name,
            "total_utterances": len(samples),
            "total_duration_hours": round(total_audio_sec / 3600.0, 3),
            "total_compute_time_sec": round(compute_time_sec, 2),
            "rtf": round(rtf, 4),
            "latency_ms_per_utt": round(avg_latency_ms, 2),
            "summary_table": summary_rows,
            "qualitative_samples": qualitative_samples,
        }

        return summary_data, detailed_rows

    def evaluate_our_model(self, limit: Optional[int] = None, batch_size: int = 16) -> Dict[str, Any]:
        """Runs evaluation for Our Multi-Dialect MoE Conformer using CTC decoding."""
        print(f"\n{'='*80}")
        print(f"[1/3] Running Frozen RESPIN Evaluation: Our Model (MoE Conformer CTC)")
        print(f"{'='*80}")

        samples = self.samples[:limit] if limit else self.samples
        model = OurASRModel(checkpoint_path=self.checkpoint_path, vocab_path=self.vocab_path, device=self.device)

        predictions = []
        t0 = time.time()

        for i in tqdm(range(0, len(samples), batch_size), desc="Our Model (CTC)"):
            batch_items = samples[i:i + batch_size]
            audio_tensors = []
            for item in batch_items:
                audio, sr = sf.read(item["wav_path"], dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audio_tensors.append(torch.from_numpy(audio))

            batch_preds = model.transcribe_batch(audio_tensors, use_beam=False)
            predictions.extend(batch_preds)

        compute_time = time.time() - t0
        print(f"[OurModel] Completed {len(samples)} utterances in {compute_time:.2f}s ({len(samples)/compute_time:.1f} utt/s)")

        summary, detailed = self._evaluate_predictions("our_model", "Our Model (MoE Conformer)", predictions, samples, compute_time)
        self._save_model_results("our_model", summary, detailed)
        return summary

    def evaluate_sravaani(self, limit: Optional[int] = None, batch_size: int = 32) -> Dict[str, Any]:
        """Runs evaluation for ARTPARK-IISc / SraVaani-1.0 using CTC decoding."""
        print(f"\n{'='*80}")
        print(f"[2/3] Running Frozen RESPIN Evaluation: SraVaani 1.0 (ARTPARK-IISc)")
        print(f"{'='*80}")

        samples = self.samples[:limit] if limit else self.samples
        model = SraVaaniModel(device=self.device)

        wav_paths = [s["wav_path"] for s in samples]
        t0 = time.time()
        predictions = []

        # Process in chunks with progress bar
        for i in tqdm(range(0, len(wav_paths), batch_size), desc="SraVaani-1.0"):
            batch_paths = wav_paths[i:i + batch_size]
            batch_preds = model.transcribe_paths(batch_paths, batch_size=batch_size)
            predictions.extend(batch_preds)

        compute_time = time.time() - t0
        print(f"[SraVaani] Completed {len(samples)} utterances in {compute_time:.2f}s ({len(samples)/compute_time:.1f} utt/s)")

        summary, detailed = self._evaluate_predictions("sravaani", "SraVaani 1.0 (ARTPARK-IISc)", predictions, samples, compute_time)
        self._save_model_results("sravaani", summary, detailed)
        return summary

    def evaluate_indic_conformer(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Runs evaluation for AI4Bharat / IndicConformer 600M Multilingual using CTC decoding."""
        print(f"\n{'='*80}")
        print(f"[3/3] Running Frozen RESPIN Evaluation: Indic Conformer 600M (AI4Bharat)")
        print(f"{'='*80}")

        samples = self.samples[:limit] if limit else self.samples
        model = IndicConformerModel(device=self.device)

        from concurrent.futures import ThreadPoolExecutor

        def _transcribe_item(s):
            try:
                return model.transcribe_single(s["wav_path"])
            except Exception:
                return ""

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            predictions = list(tqdm(executor.map(_transcribe_item, samples), total=len(samples), desc="IndicConformer (CTC)"))

        compute_time = time.time() - t0
        print(f"[IndicConformer] Completed {len(samples)} utterances in {compute_time:.2f}s ({len(samples)/compute_time:.1f} utt/s)")

        summary, detailed = self._evaluate_predictions("indic_conformer", "Indic Conformer 600M (AI4Bharat)", predictions, samples, compute_time)
        self._save_model_results("indic_conformer", summary, detailed)
        return summary

    def _save_model_results(self, model_key: str, summary: Dict[str, Any], detailed: List[Dict[str, Any]], custom_dir: Optional[Path] = None):
        """Saves summary CSV, detailed CSV, and JSON for an individual model."""
        target_dir = custom_dir if custom_dir is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        sum_csv = target_dir / f"{model_key}_respin_summary.csv"
        with open(sum_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary["summary_table"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["summary_table"])
        print(f"[Artifact] Saved: {sum_csv}")

        # Detailed CSV
        det_csv = target_dir / f"{model_key}_respin_detailed.csv"
        with open(det_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed[0].keys()))
            writer.writeheader()
            writer.writerows(detailed)
        print(f"[Artifact] Saved: {det_csv}")

        # Summary JSON
        sum_json = target_dir / f"{model_key}_respin_summary.json"
        with open(sum_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Artifact] Saved: {sum_json}")

    def generate_final_comparison_report(self, summaries: List[Dict[str, Any]]):
        """Generates unified Markdown report and master comparison CSV."""
        print(f"\n{'='*80}")
        print("               GENERATING COMPREHENSIVE BENCHMARK REPORT")
        print(f"{'='*80}")

        # Master comparison CSV
        all_rows = []
        for s in summaries:
            all_rows.extend(s["summary_table"])

        comp_csv = self.output_dir / "all_models_comparison_summary.csv"
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[Artifact] Saved master comparison CSV: {comp_csv}")

        # Unified JSON
        comp_json = self.output_dir / "benchmark_comparison_summary.json"
        with open(comp_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"[Artifact] Saved master comparison JSON: {comp_json}")

        # Master Markdown Report
        md_file = self.output_dir / "benchmark_report.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# IISc RESPIN Marathi ASR Benchmark: Model Comparison Report\n\n")
            f.write(f"- **Evaluated Test Set**: IISc RESPIN Held-Out Marathi (`meta_test_mr.json`)\n")
            f.write(f"- **Total Utterances**: {summaries[0]['total_utterances']:,} ({summaries[0]['total_duration_hours']:.2f} hours)\n")
            f.write(f"- **Evaluation Protocol**: Zero-leakage held-out test evaluation using **CTC Decoding**\n")
            f.write(f"- **Target Dialects**: D1 (Malvani/Konkan), D2 (Ahirani/Khandesh), D3 (Standard Marathi), D4 (Varhadi/Vidarbha)\n\n")

            f.write("## 1. Overall System Performance Summary (Aggregate ALL)\n\n")
            f.write("| Model Name | Decoding | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for s in summaries:
                all_stat = next(r for r in s["summary_table"] if r["dialect_id"] == "ALL")
                f.write(
                    f"| **{s['model_name']}** | CTC | "
                    f"**{all_stat['cer_percent']:.2f}%** | "
                    f"**{all_stat['wer_percent']:.2f}%** | "
                    f"{all_stat['ser_percent']:.2f}% | "
                    f"{all_stat['exact_match_acc']:.2f}% | "
                    f"{all_stat['rtf']}x | "
                    f"{all_stat['latency_ms_per_utt']:.1f} ms |\n"
                )

            f.write("\n## 2. Dialect-by-Dialect Comparative Breakdown\n\n")
            f.write("### Character Error Rate (CER %) by Dialect\n\n")
            f.write("| Dialect | Name | " + " | ".join([f"**{s['model_name']}**" for s in summaries]) + " |\n")
            f.write("| :--- | :--- | " + " | ".join([":---:" for _ in summaries]) + " |\n")

            for d in ["D1", "D2", "D3", "D4", "ALL"]:
                d_name = DIALECT_NAMES.get(d, "Aggregate")
                vals = []
                for s in summaries:
                    row = next(r for r in s["summary_table"] if r["dialect_id"] == d)
                    vals.append(f"{row['cer_percent']:.2f}%")
                f.write(f"| {d} | {d_name} | " + " | ".join(vals) + " |\n")

            f.write("\n### Word Error Rate (WER %) by Dialect\n\n")
            f.write("| Dialect | Name | " + " | ".join([f"**{s['model_name']}**" for s in summaries]) + " |\n")
            f.write("| :--- | :--- | " + " | ".join([":---:" for _ in summaries]) + " |\n")

            for d in ["D1", "D2", "D3", "D4", "ALL"]:
                d_name = DIALECT_NAMES.get(d, "Aggregate")
                vals = []
                for s in summaries:
                    row = next(r for r in s["summary_table"] if r["dialect_id"] == d)
                    vals.append(f"{row['wer_percent']:.2f}%")
                f.write(f"| {d} | {d_name} | " + " | ".join(vals) + " |\n")

            f.write("\n## 3. Qualitative Comparative Samples\n\n")
            f.write("| Dialect | Reference | " + " | ".join([f"**{s['model_name']}**" for s in summaries]) + " |\n")
            f.write("| :---: | :--- | " + " | ".join([":---" for _ in summaries]) + " |\n")

            # Collect qualitative samples by UID
            sample_uids = [q["uid"] for q in summaries[0]["qualitative_samples"][:10]]
            for uid in sample_uids:
                ref = next(q["reference"] for q in summaries[0]["qualitative_samples"] if q["uid"] == uid)
                dia = next(q["dialect"] for q in summaries[0]["qualitative_samples"] if q["uid"] == uid)
                preds = []
                for s in summaries:
                    match = next((q["prediction"] for q in s["qualitative_samples"] if q["uid"] == uid), "N/A")
                    preds.append(match)
                f.write(f"| {dia} | {ref} | " + " | ".join(preds) + " |\n")

        print(f"[Artifact] Saved comprehensive Markdown report: {md_file}\n")


    def run_kenlm_ablation(
        self,
        limit: Optional[int] = None,
        batch_size: int = 32,
        beam_width: int = 16,
        kenlm_path: str = "lm/marathi_5gram.binary",
        output_subdir: str = "kenlm_ablation",
    ) -> List[Dict[str, Any]]:
        """
        Runs rigorous controlled ablation comparing:
          1. Greedy CTC baseline (alpha = 0.0)
          2. Lexicon-only Beam Search (prefix-tree with wordlist, no LM)
          3. Current KenLM with alpha = 0.5 (default)
          4. Current KenLM with alpha = 0.15 (soft LM constraint)
        Across each dialect subset (D1, D2, D3, D4) and aggregate ALL.
        """
        out_path = Path("inference/results") / output_subdir
        out_path.mkdir(parents=True, exist_ok=True)

        samples = self.samples[:limit] if limit else self.samples
        print(f"\n{'='*80}")
        print(f"       KENLM & DECODER ABLATION STUDY ON IISc RESPIN TEST SET")
        print(f"Total utterances: {len(samples):,} | Device: {self.device} | Checkpoint: {self.checkpoint_path}")
        print(f"{'='*80}")

        model = OurASRModel(checkpoint_path=self.checkpoint_path, vocab_path=self.vocab_path, device=self.device)
        wordlist = load_marathi_wordlist(max_words=35000)
        print(f"[Ablation] Lexicon wordlist loaded: {len(wordlist):,} entries")

        # Configurations to ablate
        configs = [
            {
                "key": "greedy_ctc",
                "name": "Greedy CTC Baseline (alpha=0.0)",
                "decoder": None,
            },
            {
                "key": "lexicon_beam",
                "name": "Lexicon-Only Beam Search (No LM)",
                "decoder": MarathiBeamSearchDecoder(
                    model.tokenizer,
                    unigram_words=wordlist,
                    kenlm_path=None,
                    alpha=0.0,
                    beta=1.0,
                ),
            },
            {
                "key": "kenlm_alpha_0_5",
                "name": "KenLM 5-gram (alpha=0.5, default)",
                "decoder": MarathiBeamSearchDecoder(
                    model.tokenizer,
                    unigram_words=wordlist,
                    kenlm_path=kenlm_path,
                    alpha=0.5,
                    beta=1.5,
                ),
            },
            {
                "key": "kenlm_alpha_0_15",
                "name": "KenLM 5-gram Soft (alpha=0.15)",
                "decoder": MarathiBeamSearchDecoder(
                    model.tokenizer,
                    unigram_words=wordlist,
                    kenlm_path=kenlm_path,
                    alpha=0.15,
                    beta=1.0,
                ),
            },
        ]

        # Stage 1: Single acoustic forward pass to cache exact logits
        print(f"\n[Ablation] Phase 1: Computing acoustic model logits for {len(samples)} utterances...")
        t_ac_start = time.time()
        cached_logits = []
        for i in tqdm(range(0, len(samples), batch_size), desc="Acoustic Forward Pass"):
            batch_items = samples[i:i + batch_size]
            audio_tensors = []
            for item in batch_items:
                audio, sr = sf.read(item["wav_path"], dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audio_tensors.append(torch.from_numpy(audio).float())

            b_cur = len(audio_tensors)
            lens = torch.tensor([len(a) for a in audio_tensors], dtype=torch.long)
            max_len = lens.max().item()
            padded = torch.zeros(b_cur, max_len, dtype=torch.float32, device=self.device)
            for idx, a in enumerate(audio_tensors):
                padded[idx, :len(a)] = a.to(self.device)

            with torch.no_grad():
                ctc_dict = model.model.forward_ctc(padded, chunk_size=None)
                logits = ctc_dict["log_probs"].cpu()

            for idx in range(b_cur):
                valid_frames = max(1, (len(audio_tensors[idx]) // 160) // 4)
                cached_logits.append(logits[idx, :valid_frames].numpy())

        acoustic_time = time.time() - t_ac_start
        print(f"[Ablation] Logits computed in {acoustic_time:.2f}s ({len(samples)/acoustic_time:.1f} utt/s)")

        # Stage 2: Decode with each configuration
        ablation_summaries = []
        for cfg in configs:
            print(f"\n{'='*70}")
            print(f"[Ablation] Phase 2: Decoding with {cfg['name']}")
            print(f"{'='*70}")

            t_dec_start = time.time()
            preds = []
            for logit in tqdm(cached_logits, desc=cfg["key"]):
                if cfg["decoder"] is None:
                    tokens = np.argmax(logit, axis=-1).tolist()
                    pred = sanitize_transcript(model.tokenizer.ctc_decode(tokens))
                else:
                    pred = cfg["decoder"].decode(logit, beam_width=beam_width)
                preds.append(pred)

            decode_time = time.time() - t_dec_start
            total_time = acoustic_time + decode_time
            print(f"[Ablation] Decoding finished in {decode_time:.2f}s (Total pipeline: {total_time:.2f}s)")

            summary, detailed = self._evaluate_predictions(
                cfg["key"], cfg["name"], preds, samples, total_time
            )
            self._save_model_results(cfg["key"], summary, detailed, custom_dir=out_path)
            ablation_summaries.append(summary)

        # Stage 3: Generate unified ablation Markdown report and master CSV
        self.generate_ablation_report(ablation_summaries, out_path)
        return ablation_summaries

    def generate_ablation_report(self, summaries: List[Dict[str, Any]], out_dir: Path):
        """Generates comprehensive KenLM ablation Markdown report and master comparison CSV."""
        print(f"\n{'='*80}")
        print("         GENERATING DETAILED KENLM ABLATION REPORT")
        print(f"{'='*80}")

        # Master comparison CSV
        all_rows = []
        for s in summaries:
            all_rows.extend(s["summary_table"])

        comp_csv = out_dir / "kenlm_ablation_comparison.csv"
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[Artifact] Saved master comparison CSV: {comp_csv}")

        # Unified JSON
        comp_json = out_dir / "kenlm_ablation_summary.json"
        with open(comp_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"[Artifact] Saved master JSON: {comp_json}")

        # Master Markdown Report
        md_file = out_dir / "kenlm_ablation_report.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# IISc RESPIN KenLM & Decoding Ablation Benchmark Report\n\n")
            f.write(f"- **Evaluated Test Set**: IISc RESPIN Held-Out Marathi (`meta_test_mr.json`)\n")
            f.write(f"- **Total Utterances**: {summaries[0]['total_utterances']:,} ({summaries[0]['total_duration_hours']:.2f} hours)\n")
            f.write(f"- **Target Dialects**: D1 (Malvani/Konkan), D2 (Ahirani/Khandesh), D3 (Standard Marathi), D4 (Varhadi/Vidarbha)\n")
            f.write(f"- **Objective**: Determine whether standard 5-gram KenLM aids or degrades dialect speech recognition, and whether lexical prefix-tree decoding or soft LM constraint is superior.\n\n")

            f.write("## 1. Overall System Performance Summary (Aggregate ALL)\n\n")
            f.write("| Decoder Configuration | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for s in summaries:
                all_stat = next(r for r in s["summary_table"] if r["dialect_id"] == "ALL")
                f.write(
                    f"| **{s['model_name']}** | "
                    f"**{all_stat['cer_percent']:.2f}%** | "
                    f"**{all_stat['wer_percent']:.2f}%** | "
                    f"{all_stat['ser_percent']:.2f}% | "
                    f"{all_stat['exact_match_acc']:.2f}% | "
                    f"{all_stat['rtf']}x | "
                    f"{all_stat['latency_ms_per_utt']:.1f} ms |\n"
                )

            f.write("\n## 2. Dialect-by-Dialect Breakdown\n\n")
            f.write("### Character Error Rate (CER %) by Dialect\n\n")
            f.write("| Dialect | Name | " + " | ".join([f"**{s['model_name']}**" for s in summaries]) + " |\n")
            f.write("| :--- | :--- | " + " | ".join([":---:" for _ in summaries]) + " |\n")

            for d in ["D1", "D2", "D3", "D4", "ALL"]:
                d_name = DIALECT_NAMES.get(d, "Aggregate")
                vals = []
                for s in summaries:
                    row = next(r for r in s["summary_table"] if r["dialect_id"] == d)
                    vals.append(f"{row['cer_percent']:.2f}%")
                f.write(f"| {d} | {d_name} | " + " | ".join(vals) + " |\n")

            f.write("\n### Word Error Rate (WER %) by Dialect\n\n")
            f.write("| Dialect | Name | " + " | ".join([f"**{s['model_name']}**" for s in summaries]) + " |\n")
            f.write("| :--- | :--- | " + " | ".join([":---:" for _ in summaries]) + " |\n")

            for d in ["D1", "D2", "D3", "D4", "ALL"]:
                d_name = DIALECT_NAMES.get(d, "Aggregate")
                vals = []
                for s in summaries:
                    row = next(r for r in s["summary_table"] if r["dialect_id"] == d)
                    vals.append(f"{row['wer_percent']:.2f}%")
                f.write(f"| {d} | {d_name} | " + " | ".join(vals) + " |\n")

            # Section 3: Delta Analysis
            f.write("\n## 3. Delta Analysis: Impact of Language Model on Dialects vs Standard Marathi\n\n")
            f.write("Relative change in WER (%) compared to Greedy CTC baseline (negative = improvement, positive = degradation):\n\n")
            f.write("| Dialect | Name | Lexicon-Only Delta | KenLM (α=0.5) Delta | KenLM (α=0.15) Delta |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: |\n")

            greedy_summary = summaries[0]
            lex_summary = summaries[1]
            lm05_summary = summaries[2]
            lm015_summary = summaries[3]

            for d in ["D1", "D2", "D3", "D4", "ALL"]:
                d_name = DIALECT_NAMES.get(d, "Aggregate")
                g_wer = next(r for r in greedy_summary["summary_table"] if r["dialect_id"] == d)["wer_percent"]
                lex_wer = next(r for r in lex_summary["summary_table"] if r["dialect_id"] == d)["wer_percent"]
                lm05_wer = next(r for r in lm05_summary["summary_table"] if r["dialect_id"] == d)["wer_percent"]
                lm015_wer = next(r for r in lm015_summary["summary_table"] if r["dialect_id"] == d)["wer_percent"]

                d_lex = lex_wer - g_wer
                d_lm05 = lm05_wer - g_wer
                d_lm015 = lm015_wer - g_wer

                f.write(
                    f"| {d} | {d_name} | "
                    f"{'+' if d_lex > 0 else ''}{d_lex:.2f}% | "
                    f"{'+' if d_lm05 > 0 else ''}{d_lm05:.2f}% | "
                    f"{'+' if d_lm015 > 0 else ''}{d_lm015:.2f}% |\n"
                )

            # Section 4: Qualitative Comparative Samples
            f.write("\n## 4. Qualitative Transcription Comparison Across Decoders\n\n")
            f.write("| Dialect | Reference | " + " | ".join([f"**{s['model_name']}**" for s in summaries]) + " |\n")
            f.write("| :---: | :--- | " + " | ".join([":---" for _ in summaries]) + " |\n")

            sample_uids = [q["uid"] for q in summaries[0]["qualitative_samples"][:12]]
            for uid in sample_uids:
                ref = next(q["reference"] for q in summaries[0]["qualitative_samples"] if q["uid"] == uid)
                dia = next(q["dialect"] for q in summaries[0]["qualitative_samples"] if q["uid"] == uid)
                preds = []
                for s in summaries:
                    match = next((q["prediction"] for q in s["qualitative_samples"] if q["uid"] == uid), "N/A")
                    preds.append(match)
                f.write(f"| {dia} | {ref} | " + " | ".join(preds) + " |\n")

        print(f"[Artifact] Saved comprehensive Markdown report: {md_file}\n")


    def compile_existing_reports(self):
        """Looks for existing individual summary JSONs and generates the unified comparison report."""
        summaries = []
        for key in ["our_model", "sravaani", "indic_conformer"]:
            sum_json = self.output_dir / f"{key}_respin_summary.json"
            if sum_json.exists():
                with open(sum_json, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
        if len(summaries) >= 2:
            self.generate_final_comparison_report(summaries)
        else:
            print("[Report] Not enough completed evaluations found to generate comparison.")


def evaluate_all(limit: Optional[int] = None, batch_size: int = 32):
    pipeline = UnifiedASRPipeline()
    results = []

    # 1. Our Model
    r_our = pipeline.evaluate_our_model(limit=limit, batch_size=batch_size)
    results.append(r_our)

    # 2. SraVaani
    r_sra = pipeline.evaluate_sravaani(limit=limit, batch_size=batch_size)
    results.append(r_sra)

    # 3. Indic Conformer
    r_ind = pipeline.evaluate_indic_conformer(limit=limit)
    results.append(r_ind)

    # Generate master tables
    pipeline.generate_final_comparison_report(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Marathi ASR Inference & Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="all", choices=["all", "our_model", "sravaani", "indic_conformer", "report", "kenlm_ablation", "ablation"], help="Model or study to evaluate")
    parser.add_argument("--test_dir", type=str, default=r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr", help="Path to RESPIN test set directory")
    parser.add_argument("--meta_file", type=str, default="meta_test_mr.json", help="Metadata file name")
    parser.add_argument("--checkpoint", type=str, default="runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt", help="Path to our model checkpoint")
    parser.add_argument("--output_dir", type=str, default="inference/results/respin", help="Output directory for reports and CSVs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples for testing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batched inference")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--input_audio", type=str, default=None, help="Single audio file path for ad-hoc transcription")

    args = parser.parse_args()

    pipeline = UnifiedASRPipeline(
        test_dir=args.test_dir,
        meta_file=args.meta_file,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.model == "report":
        pipeline.compile_existing_reports()
        return

    if args.input_audio:
        print(f"\n[Single-Inference] Transcribing: {args.input_audio}")
        if args.model in ["all", "our_model"]:
            m = OurASRModel(checkpoint_path=args.checkpoint, device=args.device)
            audio, sr = sf.read(args.input_audio, dtype="float32")
            if len(audio.shape) > 1: audio = audio.mean(axis=1)
            pred = m.transcribe_batch([torch.from_numpy(audio)])[0]
            print(f"  > Our Model: {pred}")

        if args.model in ["all", "sravaani"]:
            m = SraVaaniModel(device=args.device)
            pred = m.transcribe_paths([args.input_audio])[0]
            print(f"  > SraVaani: {pred}")

        if args.model in ["all", "indic_conformer"]:
            m = IndicConformerModel(device=args.device)
            pred = m.transcribe_single(args.input_audio)
            print(f"  > Indic Conformer: {pred}")
        return

    if args.model in ["ablation", "kenlm_ablation"]:
        pipeline.run_kenlm_ablation(limit=args.limit, batch_size=args.batch_size)
        return

    results = []
    if args.model in ["all", "our_model"]:
        results.append(pipeline.evaluate_our_model(limit=args.limit, batch_size=args.batch_size))

    if args.model in ["all", "sravaani"]:
        results.append(pipeline.evaluate_sravaani(limit=args.limit, batch_size=args.batch_size))

    if args.model in ["all", "indic_conformer"]:
        results.append(pipeline.evaluate_indic_conformer(limit=args.limit))

    if len(results) > 1:
        pipeline.generate_final_comparison_report(results)


if __name__ == "__main__":
    main()
