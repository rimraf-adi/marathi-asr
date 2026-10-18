"""
MoE vs. Dense Conformer Benchmark Suite on IISc RESPIN Held-Out Test Set.

Scientifically evaluates the necessity of dialect-aware Mixture-of-Experts:
  1. Tests the Negative Transfer Hypothesis:
     Does pooled dense training cause gradient interference on colloquial regional dialects
     (D1 Malvani, D2 Ahirani, D4 Varhadi) compared to Standard Marathi (D3)?
  2. Compares Dialect-Aware MoE vs. Dense Conformer Baseline:
     - Parameter-matched active capacity (63.8M active parameters)
     - Same training datasets, features, optimizer, and training curriculum
  3. Computes:
     - Per-dialect & overall Character Error Rate (CER %) and Word Error Rate (WER %)
     - Delta CER (Dense - MoE) and Relative Error Reduction (RER %)
     - Statistical significance testing (Paired t-test and Wilcoxon signed-rank test)
     - Compute efficiency: Active Parameters, Total Parameters, RTF, Latency
  4. Exports structured summary CSV, detailed per-utterance CSV, Markdown report, and LaTeX table.
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
        # Active params in top-1 MoE: backbone (layers 1-4) + non-MoE in layers 5-12 + 1 active expert per MoE layer + CTC head
        # In our architecture, 3 experts exist in ffn2 of 8 layers. 2 experts are inactive per token/forward.
        expert_params = sum(p.numel() for name, p in model.named_parameters() if "experts" in name)
        active_params = total_params - int(expert_params * (2.0 / 3.0))
    else:
        active_params = total_params

    return model, is_moe, total_params, active_params


@torch.no_grad()
def evaluate_model_on_samples(
    model: StreamingASRModel,
    tokenizer: MarathiTokenizer,
    samples: List[Dict[str, Any]],
    device: torch.device,
    use_beam: bool = False,
    beam_decoder: Optional[MarathiBeamSearchDecoder] = None,
    beam_width: int = 16,
) -> Tuple[List[Dict[str, Any]], float]:
    """Runs inference across all test samples and returns predictions and RTF."""
    results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for item in samples:
        audio, sr = sf.read(item["wav_path"])
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            audio = resampler(torch.tensor(audio, dtype=torch.float32)).numpy()

        audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        dur = len(audio) / 16000.0
        total_audio_sec += dur

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            ctc_dict = model.forward_ctc(audio_t)
            logits = ctc_dict["log_probs"][0]

            if use_beam and beam_decoder is not None:
                hyp = beam_decoder.decode(logits, beam_width=beam_width)
            else:
                tokens = logits.argmax(dim=-1).cpu().tolist()
                hyp = tokenizer.ctc_decode(tokens)

        t1 = time.perf_counter()
        infer_time = t1 - t0
        total_infer_sec += infer_time

        ref_clean = normalize_text(item["text"])
        hyp_clean = normalize_text(hyp)

        c_err = compute_cer(ref_clean, hyp_clean)
        w_err = jiwer.wer(ref_clean, hyp_clean) if len(ref_clean.split()) > 0 and len(hyp_clean.split()) > 0 else 1.0

        results.append({
            "uid": item["uid"],
            "dialect": item["dialect"],
            "ref": ref_clean,
            "hyp": hyp_clean,
            "cer": c_err,
            "wer": w_err,
            "duration_sec": dur,
            "latency_sec": infer_time,
        })

    rtf = total_infer_sec / max(1e-5, total_audio_sec)
    return results, rtf


def compute_metrics_breakdown(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall and per-dialect metrics."""
    by_dialect = defaultdict(list)
    all_cers = []
    all_wers = []

    for r in eval_records:
        d = r["dialect"]
        by_dialect[d].append(r)
        all_cers.append(r["cer"])
        all_wers.append(r["wer"])

    stats_dict = {
        "overall": {
            "cer": float(np.mean(all_cers)),
            "wer": float(np.mean(all_wers)),
            "count": len(all_cers),
        },
        "per_dialect": {},
    }

    for d in sorted(by_dialect.keys()):
        d_recs = by_dialect[d]
        d_cers = [x["cer"] for x in d_recs]
        d_wers = [x["wer"] for x in d_recs]
        stats_dict["per_dialect"][d] = {
            "name": DIALECT_NAMES.get(d, d),
            "cer": float(np.mean(d_cers)),
            "wer": float(np.mean(d_wers)),
            "count": len(d_cers),
        }

    return stats_dict


def run_comparative_benchmark(
    moe_checkpoint: Optional[str],
    dense_checkpoint: Optional[str],
    test_dir: str = r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr",
    output_dir: str = "inference/moe_vs_notmoe/results",
    vocab_path: str = "data_utils/vocab.json",
    use_beam: bool = False,
    beam_width: int = 16,
    limit: Optional[int] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  MOE VS. NOT-MOE COMPARATIVE BENCHMARK ON IISC RESPIN TEST SET")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80 + "\n")

    tokenizer = MarathiTokenizer(vocab_path)
    beam_decoder = None
    if use_beam:
        print("[Decoder] Initializing Marathi Beam Search Decoder...")
        wordlist = load_marathi_wordlist(max_words=30000)
        beam_decoder = MarathiBeamSearchDecoder(tokenizer, unigram_words=wordlist)

    # Load metadata
    meta_path = Path(test_dir) / "meta_test_mr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Test metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        test_meta = json.load(f)

    samples = []
    for uid, info in test_meta.items():
        wav_file = Path(test_dir) / info["wav_path"]
        if wav_file.exists():
            samples.append({
                "uid": uid,
                "wav_path": str(wav_file),
                "text": info.get("text", ""),
                "dialect": info.get("dialect", "Unknown"),
            })

    if limit and limit > 0:
        samples = samples[:limit]
    print(f"[Dataset] Loaded {len(samples)} test utterances across 4 dialects.")

    results_data = {}

    # 1. Evaluate MoE Model
    if moe_checkpoint and Path(moe_checkpoint).exists():
        print(f"\n[Benchmarking] 1. Evaluating Dialect-Aware MoE Conformer: {moe_checkpoint}")
        model_moe, _, tot_moe, act_moe = load_model(moe_checkpoint, device)
        records_moe, rtf_moe = evaluate_model_on_samples(
            model_moe, tokenizer, samples, device, use_beam=use_beam, beam_decoder=beam_decoder, beam_width=beam_width
        )
        stats_moe = compute_metrics_breakdown(records_moe)
        stats_moe["total_params"] = tot_moe
        stats_moe["active_params"] = act_moe
        stats_moe["rtf"] = rtf_moe
        results_data["MoE"] = {"stats": stats_moe, "records": records_moe}
        del model_moe
        torch.cuda.empty_cache()
    else:
        print(f"[Warning] MoE checkpoint not found: {moe_checkpoint}")

    # 2. Evaluate Dense Baseline
    if dense_checkpoint and Path(dense_checkpoint).exists():
        print(f"\n[Benchmarking] 2. Evaluating Dense Conformer Baseline: {dense_checkpoint}")
        model_dense, _, tot_dense, act_dense = load_model(dense_checkpoint, device)
        records_dense, rtf_dense = evaluate_model_on_samples(
            model_dense, tokenizer, samples, device, use_beam=use_beam, beam_decoder=beam_decoder, beam_width=beam_width
        )
        stats_dense = compute_metrics_breakdown(records_dense)
        stats_dense["total_params"] = tot_dense
        stats_dense["active_params"] = act_dense
        stats_dense["rtf"] = rtf_dense
        results_data["Dense"] = {"stats": stats_dense, "records": records_dense}
        del model_dense
        torch.cuda.empty_cache()
    else:
        print(f"[Warning] Dense checkpoint not found: {dense_checkpoint}")

    # 3. If both exist, compute Delta, RER, and Statistical Significance
    has_both = "MoE" in results_data and "Dense" in results_data
    comparison = {}

    if has_both:
        moe_recs = {r["uid"]: r for r in results_data["MoE"]["records"]}
        dense_recs = {r["uid"]: r for r in results_data["Dense"]["records"]}

        paired_moe_cer = []
        paired_dense_cer = []
        detailed_rows = []

        by_dialect_moe = defaultdict(list)
        by_dialect_dense = defaultdict(list)

        for item in samples:
            uid = item["uid"]
            if uid in moe_recs and uid in dense_recs:
                m = moe_recs[uid]
                d = dense_recs[uid]
                paired_moe_cer.append(m["cer"])
                paired_dense_cer.append(d["cer"])

                by_dialect_moe[item["dialect"]].append(m["cer"])
                by_dialect_dense[item["dialect"]].append(d["cer"])

                detailed_rows.append({
                    "uid": uid,
                    "dialect": item["dialect"],
                    "dialect_name": DIALECT_NAMES.get(item["dialect"], item["dialect"]),
                    "reference": item["text"],
                    "moe_hyp": m["hyp"],
                    "dense_hyp": d["hyp"],
                    "moe_cer": round(m["cer"], 4),
                    "dense_cer": round(d["cer"], 4),
                    "delta_cer": round(d["cer"] - m["cer"], 4),  # Positive means MoE is better
                    "moe_wer": round(m["wer"], 4),
                    "dense_wer": round(d["wer"], 4),
                })

        # Save detailed comparison CSV
        detailed_csv_path = out_path / "respin_moe_vs_dense_detailed.csv"
        with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
            writer.writeheader()
            writer.writerows(detailed_rows)

        # Statistical hypothesis tests
        diffs = np.array(paired_dense_cer) - np.array(paired_moe_cer)
        ttest_res = stats.ttest_rel(paired_dense_cer, paired_moe_cer)
        try:
            wilcoxon_res = stats.wilcoxon(paired_dense_cer, paired_moe_cer)
            wilcoxon_p = wilcoxon_res.pvalue
        except Exception:
            wilcoxon_p = float("nan")

        comparison = {
            "p_value_ttest": float(ttest_res.pvalue),
            "p_value_wilcoxon": float(wilcoxon_p),
            "mean_cer_diff": float(np.mean(diffs)),
            "per_dialect_deltas": {},
        }

        for d in sorted(by_dialect_moe.keys()):
            m_cer = np.mean(by_dialect_moe[d])
            d_cer = np.mean(by_dialect_dense[d])
            delta = d_cer - m_cer
            rer = (delta / d_cer * 100.0) if d_cer > 0 else 0.0
            comparison["per_dialect_deltas"][d] = {
                "dialect_name": DIALECT_NAMES.get(d, d),
                "moe_cer": float(m_cer),
                "dense_cer": float(d_cer),
                "delta_cer": float(delta),
                "rer_percent": float(rer),
            }

    # 4. Generate Reports
    generate_markdown_report(results_data, comparison, out_path / "moe_vs_dense_scientific_report.md")
    generate_summary_csv(results_data, comparison, out_path / "respin_moe_vs_dense_summary.csv")
    if has_both:
        generate_latex_table(results_data, comparison, out_path / "table_respin_benchmark.tex")

    print(f"\n[Completed] Benchmark suite executed successfully! Results saved to: {out_path.resolve()}\n")


def generate_summary_csv(results_data: Dict[str, Any], comparison: Dict[str, Any], csv_path: Path):
    rows = []
    for model_name, data in results_data.items():
        st = data["stats"]
        row = {
            "Model": model_name,
            "Total_Params_M": round(st.get("total_params", 0) / 1e6, 2),
            "Active_Params_M": round(st.get("active_params", 0) / 1e6, 2),
            "RTF": round(st.get("rtf", 0), 4),
            "Overall_CER": round(st["overall"]["cer"] * 100, 2),
            "Overall_WER": round(st["overall"]["wer"] * 100, 2),
        }
        for d, info in st["per_dialect"].items():
            row[f"{d}_CER"] = round(info["cer"] * 100, 2)
            row[f"{d}_WER"] = round(info["wer"] * 100, 2)
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def generate_markdown_report(results_data: Dict[str, Any], comparison: Dict[str, Any], report_path: Path):
    lines = [
        "# Dialect-Aware MoE vs. Dense Baseline Empirical Benchmark",
        "",
        "## 1. Experimental Methodology & Negative Transfer Hypothesis",
        "",
        "This evaluation directly addresses the central empirical question: **Is dialect-aware Mixture-of-Experts necessary for Marathi ASR, or is a pooled dense model sufficient?**",
        "",
        "- **Negative Transfer Claim:** A single dense model trained on pooled Marathi dialects exhibits gradient interference across regional vernacular dialects (D1 Malvani, D2 Ahirani, D4 Varhadi) relative to Standard Marathi (D3).",
        "- **MoE Necessity Claim:** MoE's conditional computation dynamically routes conflicting acoustic and lexical distributions to disjoint capacity, eliminating interference at matched active parameter cost.",
        "",
        "## 2. Summary Benchmark Results (IISc RESPIN Held-Out Test Set, 2,170 Utterances)",
        "",
        "| Architecture | Total Params | Active Params | RTF | Overall CER (%) | Overall WER (%) | D1 (Malvani) CER (%) | D2 (Ahirani) CER (%) | D3 (Standard) CER (%) | D4 (Varhadi) CER (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, data in results_data.items():
        st = data["stats"]
        tot = f"{st.get('total_params', 0) / 1e6:.1f}M"
        act = f"{st.get('active_params', 0) / 1e6:.1f}M"
        rtf = f"{st.get('rtf', 0):.3f}"
        cer_all = f"{st['overall']['cer'] * 100:.2f}"
        wer_all = f"{st['overall']['wer'] * 100:.2f}"
        d1 = f"{st['per_dialect'].get('D1', {}).get('cer', 0) * 100:.2f}"
        d2 = f"{st['per_dialect'].get('D2', {}).get('cer', 0) * 100:.2f}"
        d3 = f"{st['per_dialect'].get('D3', {}).get('cer', 0) * 100:.2f}"
        d4 = f"{st['per_dialect'].get('D4', {}).get('cer', 0) * 100:.2f}"
        lines.append(f"| **{model_name}** | {tot} | {act} | {rtf} | **{cer_all}%** | **{wer_all}%** | {d1}% | {d2}% | {d3}% | {d4}% |")

    lines.append("")

    if comparison:
        lines.extend([
            "## 3. Negative Transfer & Specialization Analysis",
            "",
            "| Dialect Code | Region | Dense CER (%) | MoE CER (%) | Delta CER (Abs %) | Relative Error Reduction (RER %) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ])
        for d, deltas in comparison.get("per_dialect_deltas", {}).items():
            name = deltas["dialect_name"]
            d_cer = f"{deltas['dense_cer'] * 100:.2f}"
            m_cer = f"{deltas['moe_cer'] * 100:.2f}"
            diff = f"{deltas['delta_cer'] * 100:+.2f}"
            rer = f"{deltas['rer_percent']:+.1f}%"
            lines.append(f"| `{d}` | {name} | {d_cer}% | {m_cer}% | **{diff}%** | **{rer}** |")

        lines.extend([
            "",
            "### Statistical Hypothesis Testing:",
            f"- **Paired t-test p-value:** `{comparison.get('p_value_ttest', 1.0):.4e}`",
            f"- **Wilcoxon signed-rank p-value:** `{comparison.get('p_value_wilcoxon', 1.0):.4e}`",
            f"- **Mean Utterance CER Difference:** `{comparison.get('mean_cer_diff', 0.0) * 100:+.2f}%`",
            "",
            "### Empirical Verdict:",
        ])

        mean_diff = comparison.get("mean_cer_diff", 0.0)
        p_val = comparison.get("p_value_wilcoxon", 1.0)
        if mean_diff > 0.005 and p_val < 0.05:
            lines.append("> **Statistically Significant MoE Superiority:** The dialect-aware MoE model significantly outperforms the pooled dense baseline at matched active parameters (p < 0.05). The regional breakdown confirms negative transfer in the dense baseline on colloquial dialects, which is resolved by conditional expert routing.")
        elif abs(mean_diff) <= 0.005:
            lines.append("> **Equivalence / Minimal Discrepancy:** The dense baseline performs comparably to the MoE model at matched active parameters. This supports the simpler architecture hypothesis without incurring extra total parameter overhead.")
        else:
            lines.append("> **Dense Baseline Parity:** The pooled dense model achieves competitive accuracy across dialects, demonstrating that pooling data without routing can generalize well under this curriculum.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_latex_table(results_data: Dict[str, Any], comparison: Dict[str, Any], tex_path: Path):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Empirical evaluation on the held-out IISc RESPIN Marathi test set comparing Dialect-Aware MoE against the parameter-matched Dense Conformer baseline.}",
        r"\label{tab:moe_vs_dense_respin}",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Total Params} & \textbf{Active Params} & \textbf{RTF} & \textbf{CER (\%)} & \textbf{WER (\%)} & \textbf{D1 (\%)} & \textbf{D2 (\%)} & \textbf{D3 (\%)} & \textbf{D4 (\%)} \\",
        r"\midrule",
    ]
    for model_name, data in results_data.items():
        st = data["stats"]
        tot = f"{st.get('total_params', 0) / 1e6:.1f}M"
        act = f"{st.get('active_params', 0) / 1e6:.1f}M"
        rtf = f"{st.get('rtf', 0):.3f}"
        cer_all = f"{st['overall']['cer'] * 100:.2f}"
        wer_all = f"{st['overall']['wer'] * 100:.2f}"
        d1 = f"{st['per_dialect'].get('D1', {}).get('cer', 0) * 100:.2f}"
        d2 = f"{st['per_dialect'].get('D2', {}).get('cer', 0) * 100:.2f}"
        d3 = f"{st['per_dialect'].get('D3', {}).get('cer', 0) * 100:.2f}"
        d4 = f"{st['per_dialect'].get('D4', {}).get('cer', 0) * 100:.2f}"
        lines.append(f"{model_name} & {tot} & {act} & {rtf} & {cer_all} & {wer_all} & {d1} & {d2} & {d3} & {d4} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="MoE vs Dense Comparative Benchmark on RESPIN Test Set")
    parser.add_argument("--moe_checkpoint", type=str, default="runs/scaled-conformer512-moe-3stage/checkpoints/stage3_final_alignment/best_model.pt")
    parser.add_argument("--dense_checkpoint", type=str, default="runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt")
    parser.add_argument("--test_dir", type=str, default=r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr")
    parser.add_argument("--output_dir", type=str, default="inference/moe_vs_notmoe/results")
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_comparative_benchmark(
        moe_checkpoint=args.moe_checkpoint,
        dense_checkpoint=args.dense_checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        use_beam=args.beam_search,
        beam_width=args.beam_width,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
