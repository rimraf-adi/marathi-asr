"""
GRPO vs. Vanilla SFT Benchmark Suite on IISc RESPIN Held-Out Test Set.

Ablation Study: Direct Sequence-Level CER/WER Optimization (GRPO) vs. Vanilla Supervised CTC Fine-Tuning (SFT).

Evaluates:
  1. Vanilla SFT:
     - Calibrated via supervised CTC loss on the multi-source dataset pool
       (100% RESPIN 122.6k utts + OpenSLR64 + Kathbath Clean/Noisy + Pretrain Stream)
  2. GRPO Polish:
     - Group Relative Policy Optimization minimizing sequence edit distance / CER
       with KL divergence constraint against the reference model and auxiliary CTC anchor
  3. Metrics:
     - Per-dialect & overall CER (%) and WER (%)
     - Delta CER (SFT - GRPO) and Relative Error Reduction (RER %)
     - Paired statistical significance testing (Paired t-test and Wilcoxon signed-rank test)
     - RTF (Real-Time Factor) & Latency
  4. Exports:
     - Detailed per-utterance predictions CSV
     - Per-dialect summary CSV
     - Publication-ready LaTeX table
     - Comprehensive Markdown ablation report
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import soundfile as sf
import numpy as np
import torch
import jiwer
from scipy import stats

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer
from decoding.beam_search_decoder import MarathiBeamSearchDecoder, load_marathi_wordlist
from model.asr_model import StreamingASRModel
from moe.upcycling import load_moe_model

DIALECT_NAMES = {
    "D1": "Malvani / Konkan",
    "D2": "Ahirani / Khandesh",
    "D3": "Standard Marathi",
    "D4": "Varhadi / Vidarbha",
}

PUNCTUATION_CHARS = ["?", "!", ",", ".", ":", ";", '"', "'", "‘", "’", "“", "”", "—", "-", "।", "(", ")", "[", "]", "{", "}"]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = sanitize_transcript(text)
    for p in PUNCTUATION_CHARS:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[StreamingASRModel, bool, int, int]:
    """Loads either MoE or Dense Conformer, returns (model, is_moe, total_params, active_params)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    is_moe = any("ffn2.experts" in k for k in state_dict.keys())

    if is_moe:
        model = load_moe_model(checkpoint_path, device=device)
    else:
        model_cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
        d_model = model_cfg.get("d_model", 512)
        n_heads = model_cfg.get("n_heads", 8 if d_model == 512 else 4)
        ffn_expansion = model_cfg.get("ffn_expansion", 8)
        num_layers = model_cfg.get("num_layers", 12)
        model = StreamingASRModel(
            feat_dim=80,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            conv_kernel_size=31,
            ffn_expansion=ffn_expansion,
            dropout=0.0,
            exit_layers=[num_layers],
            enable_reconstruction_head=False,
            vocab_size=105,
        ).to(device)
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("reconstruction_head."):
                continue
            if k.startswith("backbone."):
                backbone_dict[k.replace("backbone.", "")] = v
            else:
                backbone_dict[k] = v
        model.load_state_dict(backbone_dict, strict=False)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())

    if is_moe:
        expert_params_per_layer = sum(p.numel() for p in model.encoder.layers[4].ffn2.experts[0].parameters())
        num_moe_layers = len([l for l in model.encoder.layers if hasattr(l, "ffn2") and hasattr(l.ffn2, "experts")])
        inactive_params = num_moe_layers * 2 * expert_params_per_layer
        active_params = total_params - inactive_params
    else:
        active_params = total_params

    return model, is_moe, total_params, active_params


def load_respin_test_manifest(manifest_path: str, data_dir: str) -> List[Dict[str, Any]]:
    test_samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rel_path = item.get("audio_filepath", "")
            wav_path = os.path.join(data_dir, rel_path) if not os.path.isabs(rel_path) else rel_path

            # District to Dialect mapping
            dialect = item.get("dialect", None)
            if not dialect:
                district = item.get("district", "").strip().lower()
                if district in ["ratnagiri", "sindhudurg"]:
                    dialect = "D1"
                elif district in ["jalgaon", "dhule", "nandurbar"]:
                    dialect = "D2"
                elif district in ["pune", "mumbai", "satara", "solapur", "kolhapur", "nashik", "aurangabad", "ahmednagar", "sangli"]:
                    dialect = "D3"
                elif district in ["nagpur", "amravati", "akola", "wardha", "yavatmal", "buldhana", "washim", "chandrapur", "gadchiroli", "bhandara", "gondia"]:
                    dialect = "D4"
                else:
                    dialect = "D3"

            text = normalize_text(item.get("text", ""))
            if not text:
                continue

            test_samples.append({
                "wav_path": wav_path,
                "text": text,
                "duration": item.get("duration", 0.0),
                "dialect": dialect,
                "speaker_id": item.get("speaker_id", "unknown"),
            })
    return test_samples


def decode_utterance(
    model: StreamingASRModel,
    decoder: MarathiBeamSearchDecoder,
    audio_path: str,
    device: torch.device,
) -> Tuple[str, float]:
    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    t0 = time.time()
    t_audio = torch.from_numpy(audio).float().unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            ctc_dict = model.forward_ctc(t_audio, chunk_size=None)
            log_probs = ctc_dict["log_probs"][0]  # (T, V)

    hyp_tokens = decoder.decode(log_probs)
    hyp_text = decoder.tokenizer.ctc_decode(hyp_tokens)
    hyp_text = normalize_text(hyp_text)
    latency = time.time() - t0
    return hyp_text, latency


def run_benchmark(
    sft_ckpt: str,
    grpo_ckpt: str,
    output_dir: str,
    test_manifest: str = "data/vistaar_benchmarks/IISc_RESPIN_test_mr/manifest.json",
    test_dir: str = "data/vistaar_benchmarks/IISc_RESPIN_test_mr",
    wordlist_path: str = "data_utils/marathi_wordlist_300k.txt",
    beam_width: int = 16,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("   GRPO VS. VANILLA SFT ABLATION BENCHMARK (IISc RESPIN TEST SET)")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"SFT Checkpoint : {sft_ckpt}")
    print(f"GRPO Checkpoint: {grpo_ckpt}")
    print(f"Beam Width     : {beam_width}")
    print(f"Output Dir     : {out_path.resolve()}\n")

    # Load Tokenizer & Wordlist
    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    wordlist = load_marathi_wordlist(wordlist_path)
    decoder = MarathiBeamSearchDecoder(
        tokenizer=tokenizer,
        wordlist=wordlist,
        beam_width=beam_width,
        lm_weight=0.3,
        word_score=1.0,
    )

    # 1. Load Models
    print("[1/4] Loading Vanilla SFT Model...")
    sft_model, sft_is_moe, sft_total_p, sft_active_p = load_model(sft_ckpt, device)
    print(f"  -> Total: {sft_total_p / 1e6:.2f}M | Active: {sft_active_p / 1e6:.2f}M | MoE: {sft_is_moe}")

    print("[2/4] Loading GRPO Polish Model...")
    grpo_model, grpo_is_moe, grpo_total_p, grpo_active_p = load_model(grpo_ckpt, device)
    print(f"  -> Total: {grpo_total_p / 1e6:.2f}M | Active: {grpo_active_p / 1e6:.2f}M | MoE: {grpo_is_moe}")

    # 2. Ingest Test Manifest
    print(f"\n[3/4] Ingesting frozen test set: {test_manifest}")
    samples = load_respin_test_manifest(test_manifest, test_dir)
    total_audio_sec = sum(s["duration"] for s in samples)
    print(f"  -> Loaded {len(samples):,} utterances ({total_audio_sec / 3600.0:.2f} hours)")

    # Dialect sample count check
    dialect_counts = defaultdict(int)
    for s in samples:
        dialect_counts[s["dialect"]] += 1
    for d_code, count in sorted(dialect_counts.items()):
        print(f"     • {d_code} ({DIALECT_NAMES.get(d_code, 'Unknown')}): {count} utterances")

    # 3. Decoding Loop
    print("\n[4/4] Commencing Evaluation on Frozen RESPIN Test Set...")
    results = []
    per_dialect_sft = defaultdict(lambda: {"cer": [], "wer": [], "lat": [], "dur": []})
    per_dialect_grpo = defaultdict(lambda: {"cer": [], "wer": [], "lat": [], "dur": []})

    t_eval_start = time.time()
    for idx, s in enumerate(samples):
        ref = s["text"]
        dur = s["duration"]
        dialect = s["dialect"]

        # SFT Decode
        hyp_sft, lat_sft = decode_utterance(sft_model, decoder, s["wav_path"], device)
        cer_sft = compute_cer(ref, hyp_sft)
        wer_sft = jiwer.wer(ref, hyp_sft) if len(ref.split()) > 0 else (0.0 if not hyp_sft else 1.0)

        # GRPO Decode
        hyp_grpo, lat_grpo = decode_utterance(grpo_model, decoder, s["wav_path"], device)
        cer_grpo = compute_cer(ref, hyp_grpo)
        wer_grpo = jiwer.wer(ref, hyp_grpo) if len(ref.split()) > 0 else (0.0 if not hyp_grpo else 1.0)

        # Record
        per_dialect_sft[dialect]["cer"].append(cer_sft)
        per_dialect_sft[dialect]["wer"].append(wer_sft)
        per_dialect_sft[dialect]["lat"].append(lat_sft)
        per_dialect_sft[dialect]["dur"].append(dur)

        per_dialect_grpo[dialect]["cer"].append(cer_grpo)
        per_dialect_grpo[dialect]["wer"].append(wer_grpo)
        per_dialect_grpo[dialect]["lat"].append(lat_grpo)
        per_dialect_grpo[dialect]["dur"].append(dur)

        results.append({
            "idx": idx,
            "dialect": dialect,
            "dialect_name": DIALECT_NAMES.get(dialect, "Unknown"),
            "audio_path": s["wav_path"],
            "duration": dur,
            "reference": ref,
            "sft_hyp": hyp_sft,
            "sft_cer": cer_sft,
            "sft_wer": wer_sft,
            "sft_lat": lat_sft,
            "grpo_hyp": hyp_grpo,
            "grpo_cer": cer_grpo,
            "grpo_wer": wer_grpo,
            "grpo_lat": lat_grpo,
            "delta_cer": cer_sft - cer_grpo,
            "delta_wer": wer_sft - wer_grpo,
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == len(samples):
            sft_cur_cer = np.mean([r["sft_cer"] for r in results]) * 100
            grpo_cur_cer = np.mean([r["grpo_cer"] for r in results]) * 100
            print(f"  [{idx + 1:4d}/{len(samples)}] | SFT CER: {sft_cur_cer:.2f}% | GRPO CER: {grpo_cur_cer:.2f}% | ΔCER: {sft_cur_cer - grpo_cur_cer:+.2f}%")

    total_eval_time = time.time() - t_eval_start

    # Export Per-Utterance Detailed CSV
    pred_csv = out_path / "grpo_vs_sft_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "idx", "dialect", "dialect_name", "duration", "reference",
            "sft_hyp", "sft_cer", "sft_wer", "sft_lat",
            "grpo_hyp", "grpo_cer", "grpo_wer", "grpo_lat",
            "delta_cer", "delta_wer"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] Per-utterance predictions: {pred_csv.resolve()}")

    # 4. Statistical Analysis & Aggregation
    summary_rows = []
    dialects_sorted = ["D1", "D2", "D3", "D4"]

    all_sft_cer = [r["sft_cer"] for r in results]
    all_grpo_cer = [r["grpo_cer"] for r in results]
    all_sft_wer = [r["sft_wer"] for r in results]
    all_grpo_wer = [r["grpo_wer"] for r in results]

    for d in dialects_sorted:
        s_cers = per_dialect_sft[d]["cer"]
        g_cers = per_dialect_grpo[d]["cer"]
        s_wers = per_dialect_sft[d]["wer"]
        g_wers = per_dialect_grpo[d]["wer"]
        d_dur = sum(per_dialect_sft[d]["dur"])
        d_sft_lat = sum(per_dialect_sft[d]["lat"])
        d_grpo_lat = sum(per_dialect_grpo[d]["lat"])

        s_cer_m = np.mean(s_cers) * 100
        g_cer_m = np.mean(g_cers) * 100
        s_wer_m = np.mean(s_wers) * 100
        g_wer_m = np.mean(g_wers) * 100

        delta_cer = s_cer_m - g_cer_m
        rer_cer = (delta_cer / s_cer_m * 100) if s_cer_m > 0 else 0.0

        # Statistical tests on paired utterances
        t_stat, p_val_t = stats.ttest_rel(s_cers, g_cers)
        try:
            w_stat, p_val_w = stats.wilcoxon(s_cers, g_cers)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

        summary_rows.append({
            "dialect": d,
            "dialect_name": DIALECT_NAMES[d],
            "count": len(s_cers),
            "hours": round(d_dur / 3600.0, 2),
            "sft_cer": round(s_cer_m, 2),
            "grpo_cer": round(g_cer_m, 2),
            "delta_cer": round(delta_cer, 2),
            "rer_cer_pct": round(rer_cer, 2),
            "sft_wer": round(s_wer_m, 2),
            "grpo_wer": round(g_wer_m, 2),
            "sft_rtf": round(d_sft_lat / max(d_dur, 1e-4), 4),
            "grpo_rtf": round(d_grpo_lat / max(d_dur, 1e-4), 4),
            "p_val_paired_t": float(p_val_t),
            "p_val_wilcoxon": float(p_val_w),
        })

    # Overall Aggregate
    all_sft_cer_m = np.mean(all_sft_cer) * 100
    all_grpo_cer_m = np.mean(all_grpo_cer) * 100
    all_sft_wer_m = np.mean(all_sft_wer) * 100
    all_grpo_wer_m = np.mean(all_grpo_wer) * 100
    tot_delta_cer = all_sft_cer_m - all_grpo_cer_m
    tot_rer_cer = (tot_delta_cer / all_sft_cer_m * 100) if all_sft_cer_m > 0 else 0.0

    all_t_stat, all_p_t = stats.ttest_rel(all_sft_cer, all_grpo_cer)
    try:
        all_w_stat, all_p_w = stats.wilcoxon(all_sft_cer, all_grpo_cer)
    except Exception:
        all_w_stat, all_p_w = 0.0, 1.0

    total_sft_lat = sum(r["sft_lat"] for r in results)
    total_grpo_lat = sum(r["grpo_lat"] for r in results)

    summary_rows.append({
        "dialect": "OVERALL",
        "dialect_name": "Aggregate (All Dialects)",
        "count": len(results),
        "hours": round(total_audio_sec / 3600.0, 2),
        "sft_cer": round(all_sft_cer_m, 2),
        "grpo_cer": round(all_grpo_cer_m, 2),
        "delta_cer": round(tot_delta_cer, 2),
        "rer_cer_pct": round(tot_rer_cer, 2),
        "sft_wer": round(all_sft_wer_m, 2),
        "grpo_wer": round(all_grpo_wer_m, 2),
        "sft_rtf": round(total_sft_lat / max(total_audio_sec, 1e-4), 4),
        "grpo_rtf": round(total_grpo_lat / max(total_audio_sec, 1e-4), 4),
        "p_val_paired_t": float(all_p_t),
        "p_val_wilcoxon": float(all_p_w),
    })

    # Export Summary CSV
    summary_csv = out_path / "grpo_vs_sft_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Saved] Summary CSV: {summary_csv.resolve()}")

    # 5. Export Markdown Report
    report_md = out_path / "grpo_vs_sft_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# GRPO vs. Vanilla SFT Ablation Study Report\n\n")
        f.write(f"- **Evaluation Dataset:** IISc RESPIN Official Frozen Test Set ({len(results):,} utterances, {total_audio_sec / 3600.0:.2f} hours)\n")
        f.write(f"- **Beam Search Width:** {beam_width}\n")
        f.write(f"- **SFT Checkpoint:** `{sft_ckpt}`\n")
        f.write(f"- **GRPO Checkpoint:** `{grpo_ckpt}`\n\n")

        f.write("## Dialect-by-Dialect Breakdown\n\n")
        f.write("| Dialect | Name | Utterances | Hours | SFT CER (%) | GRPO CER (%) | ΔCER (%) | RER (%) | SFT WER (%) | GRPO WER (%) | Paired t p-value | Wilcoxon p-value |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(f"| {row['dialect']} | {row['dialect_name']} | {row['count']} | {row['hours']} | "
                    f"{row['sft_cer']:.2f}% | {row['grpo_cer']:.2f}% | {row['delta_cer']:+.2f}% | {row['rer_cer_pct']:+.1f}% | "
                    f"{row['sft_wer']:.2f}% | {row['grpo_wer']:.2f}% | {row['p_val_paired_t']:.4e} | {row['p_val_wilcoxon']:.4e} |\n")

        f.write("\n## Ablation Summary & Findings\n\n")
        if tot_delta_cer > 0:
            f.write(f"✅ **GRPO Polish outperforms Vanilla SFT:** Achieving an overall CER of **{all_grpo_cer_m:.2f}%** vs. **{all_sft_cer_m:.2f}%** (Relative Error Reduction of **{tot_rer_cer:+.2f}%**, p-value: {all_p_t:.2e}).\n\n")
        elif tot_delta_cer < 0:
            f.write(f"⚠️ **Vanilla SFT maintains lower CER than GRPO Polish:** Overall CER **{all_sft_cer_m:.2f}%** vs. **{all_grpo_cer_m:.2f}%** (ΔCER: {tot_delta_cer:+.2f}%).\n\n")
        else:
            f.write("⚖️ **Parity observed between GRPO Polish and Vanilla SFT.**\n\n")

    print(f"[Saved] Markdown Report: {report_md.resolve()}")

    # 6. Export Publication LaTeX Table
    latex_tex = out_path / "grpo_vs_sft_table.tex"
    with open(latex_tex, "w", encoding="utf-8") as f:
        f.write("% Publication-ready LaTeX Table: GRPO vs Vanilla SFT Ablation\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study: Group Relative Policy Optimization (GRPO) vs. Vanilla Supervised Fine-Tuning (SFT) on the IISc RESPIN Frozen Marathi Test Set. Evaluation performed with 16-beam width beam search.}\n")
        f.write("\\label{tab:grpo_vs_sft}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llcccccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Dialect} & \\textbf{Region} & \\textbf{Hours} & \\multicolumn{2}{c}{\\textbf{CER (\\%)}} & \\textbf{$\\Delta$CER} & \\textbf{RER} & \\multicolumn{2}{c}{\\textbf{WER (\\%)}} & \\textbf{Significance} \\\\\n")
        f.write(" & & & \\textbf{Vanilla SFT} & \\textbf{GRPO} & (SFT $-$ GRPO) & (\\%) & \\textbf{Vanilla SFT} & \\textbf{GRPO} & ($p$-value) \\\\\n")
        f.write("\\midrule\n")
        for row in summary_rows[:-1]:
            p_str = "$p < 0.001$" if row["p_val_paired_t"] < 0.001 else f"$p = {row['p_val_paired_t']:.3f}$"
            f.write(f"{row['dialect']} & {row['dialect_name']} & {row['hours']} & "
                    f"{row['sft_cer']:.2f} & \\textbf{{{row['grpo_cer']:.2f}}} & {row['delta_cer']:+.2f} & {row['rer_cer_pct']:+.1f}\\% & "
                    f"{row['sft_wer']:.2f} & \\textbf{{{row['grpo_wer']:.2f}}} & {p_str} \\\\\n")
        f.write("\\midrule\n")
        tot = summary_rows[-1]
        p_tot_str = "$p < 0.001$" if tot["p_val_paired_t"] < 0.001 else f"$p = {tot['p_val_paired_t']:.3f}$"
        f.write(f"\\textbf{{Overall}} & \\textbf{{Aggregate}} & \\textbf{{{tot['hours']}}} & "
                f"\\textbf{{{tot['sft_cer']:.2f}}} & \\textbf{{{tot['grpo_cer']:.2f}}} & \\textbf{{{tot['delta_cer']:+.2f}}} & \\textbf{{{tot['rer_cer_pct']:+.1f}\\%}} & "
                f"\\textbf{{{tot['sft_wer']:.2f}}} & \\textbf{{{tot['grpo_wer']:.2f}}} & {p_tot_str} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"[Saved] LaTeX Table: {latex_tex.resolve()}")

    print("\n" + "=" * 80)
    print("   GRPO VS. SFT BENCHMARK COMPLETE!")
    print(f"   Vanilla SFT Overall CER: {all_sft_cer_m:.2f}% | WER: {all_sft_wer_m:.2f}%")
    print(f"   GRPO Polish Overall CER: {all_grpo_cer_m:.2f}% | WER: {all_grpo_wer_m:.2f}%")
    print(f"   Delta CER              : {tot_delta_cer:+.2f}% (RER: {tot_rer_cer:+.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO vs Vanilla SFT Benchmark on RESPIN Test Set")
    parser.add_argument("--sft_checkpoint", type=str, required=True, help="Path to Vanilla SFT checkpoint")
    parser.add_argument("--grpo_checkpoint", type=str, required=True, help="Path to GRPO Polish checkpoint")
    parser.add_argument("--output_dir", type=str, default="inference/grpo_vs_sft/results", help="Directory to save benchmark results")
    parser.add_argument("--test_manifest", type=str, default="data/vistaar_benchmarks/IISc_RESPIN_test_mr/manifest.json")
    parser.add_argument("--test_dir", type=str, default="data/vistaar_benchmarks/IISc_RESPIN_test_mr")
    parser.add_argument("--beam_width", type=int, default=16)
    args = parser.parse_args()

    run_benchmark(
        sft_ckpt=args.sft_checkpoint,
        grpo_ckpt=args.grpo_checkpoint,
        output_dir=args.output_dir,
        test_manifest=args.test_manifest,
        test_dir=args.test_dir,
        beam_width=args.beam_width,
    )
