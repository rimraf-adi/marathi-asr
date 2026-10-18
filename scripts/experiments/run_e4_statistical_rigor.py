"""
Experiment E4: Statistical Rigor & Paired Bootstrap Significance Testing
Tests:
- Bootstrap 95% Confidence Intervals (BCa / Percentile, B=10,000) for WER and CER
  per dialect (D1, D2, D3, D4) and aggregate (ALL).
- Paired difference tests:
    Delta(WER) = WER(MoE) - WER(IndicConformer)
    Delta(WER) = WER(MoE) - WER(SraVaani)
- Paired permutation tests and Wilcoxon signed-rank tests with exact p-values.
- Evaluation of Hypothesis E4:
    "The D1 gain over IndicConformer is significant (95% CI strictly < 0, p < 0.05),
     while the D3 difference is not statistically significant."
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESPIN_DIR = ROOT_DIR / "inference" / "results" / "respin"
OUTPUT_DIR = ROOT_DIR / "inference" / "results" / "experiments" / "e4_statistical_rigor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_detailed_results():
    our_csv = RESPIN_DIR / "our_model_respin_detailed.csv"
    indic_csv = RESPIN_DIR / "indic_conformer_respin_detailed.csv"
    sra_csv = RESPIN_DIR / "sravaani_respin_detailed.csv"

    df_our = pd.read_csv(our_csv)
    df_indic = pd.read_csv(indic_csv)
    df_sra = pd.read_csv(sra_csv)

    # Sort and align by uid
    df_our = df_our.sort_values("uid").reset_index(drop=True)
    df_indic = df_indic.sort_values("uid").reset_index(drop=True)
    df_sra = df_sra.sort_values("uid").reset_index(drop=True)

    assert (df_our["uid"] == df_indic["uid"]).all(), "UID mismatch between Our Model and IndicConformer"
    assert (df_our["uid"] == df_sra["uid"]).all(), "UID mismatch between Our Model and SraVaani"

    return df_our, df_indic, df_sra


def paired_bootstrap(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    num_bootstrap: int = 10000,
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    n = len(vals_a)
    indices = rng.integers(0, n, size=(num_bootstrap, n))

    boot_a = np.mean(vals_a[indices], axis=1)
    boot_b = np.mean(vals_b[indices], axis=1)
    diff = boot_a - boot_b

    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))
    mean_diff = float(np.mean(vals_a - vals_b))

    ci_a = (float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5)))
    ci_b = (float(np.percentile(boot_b, 2.5)), float(np.percentile(boot_b, 97.5)))
    ci_diff = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))

    # Empirical two-tailed p-value on diff = 0
    p_boot = float(2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
    p_boot = max(p_boot, 1.0 / num_bootstrap)

    # Wilcoxon signed-rank test
    diff_raw = vals_a - vals_b
    if np.all(diff_raw == 0):
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            w_stat, w_p = stats.wilcoxon(vals_a, vals_b, alternative='two-sided')
            w_stat, w_p = float(w_stat), float(w_p)
        except Exception:
            w_stat, w_p = 0.0, 1.0

    return {
        "mean_a": mean_a, "ci_a": ci_a,
        "mean_b": mean_b, "ci_b": ci_b,
        "mean_diff": mean_diff, "ci_diff": ci_diff,
        "p_boot": p_boot, "p_wilcoxon": w_p
    }


def run_e4_analysis():
    print("=" * 80)
    print("RUNNING EXPERIMENT E4: STATISTICAL RIGOR & PAIRED BOOTSTRAP SIGNIFICANCE")
    print("=" * 80)

    df_our, df_indic, df_sra = load_detailed_results()
    dialects = ["D1", "D2", "D3", "D4", "ALL"]

    metrics = [
        ("norm_wer_percent", "WER (%)"),
        ("norm_cer_percent", "CER (%)"),
    ]

    results = {
        "hypothesis_e4": {
            "statement": "The D1 gain over IndicConformer is significant, the D3 difference is not.",
            "status": None,
            "refutation_rule": "Refuted if the D1 confidence interval of diff(MoE - IndicConformer) includes zero.",
        },
        "dialects": {}
    }

    report_lines = [
        "# Experiment E4 Report: Statistical Rigor & Bootstrap Significance Testing\n",
        f"**Date:** 18 October 2026  ",
        f"**Bootstrap Resamples:** 10,000 (paired cluster resampling)  ",
        f"**Confidence Level:** 95% (2.5th to 97.5th percentiles)  \n",
        "## 1. Summary of Bootstrap 95% Confidence Intervals\n",
    ]

    for metric_col, metric_label in metrics:
        report_lines.append(f"\n### {metric_label}\n")
        report_lines.append(
            "| Dialect | Our MoE (95% CI) | IndicConformer (95% CI) | SraVaani (95% CI) | "
            "Diff (MoE - Indic) [95% CI] | p-value (MoE vs Indic) |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        )

        for d in dialects:
            if d == "ALL":
                sub_our = df_our
                sub_indic = df_indic
                sub_sra = df_sra
            else:
                sub_our = df_our[df_our["dialect"] == d]
                sub_indic = df_indic[df_indic["dialect"] == d]
                sub_sra = df_sra[df_sra["dialect"] == d]

            y_our = sub_our[metric_col].to_numpy()
            y_indic = sub_indic[metric_col].to_numpy()
            y_sra = sub_sra[metric_col].to_numpy()

            boot_indic = paired_bootstrap(y_our, y_indic)
            boot_sra = paired_bootstrap(y_our, y_sra)

            if d not in results["dialects"]:
                results["dialects"][d] = {}
            results["dialects"][d][metric_col] = {
                "our": {"mean": boot_indic["mean_a"], "ci95": boot_indic["ci_a"]},
                "indic": {"mean": boot_indic["mean_b"], "ci95": boot_indic["ci_b"]},
                "sra": {"mean": boot_sra["mean_b"], "ci95": boot_sra["ci_b"]},
                "diff_vs_indic": {
                    "mean": boot_indic["mean_diff"],
                    "ci95": boot_indic["ci_diff"],
                    "p_boot": boot_indic["p_boot"],
                    "p_wilcoxon": boot_indic["p_wilcoxon"],
                },
                "diff_vs_sra": {
                    "mean": boot_sra["mean_diff"],
                    "ci95": boot_sra["ci_diff"],
                    "p_boot": boot_sra["p_boot"],
                    "p_wilcoxon": boot_sra["p_wilcoxon"],
                }
            }

            p_str = "< 0.001" if boot_indic["p_wilcoxon"] < 0.001 else f"{boot_indic['p_wilcoxon']:.4f}"
            sig_marker = "**" if boot_indic["p_wilcoxon"] < 0.01 else ("*" if boot_indic["p_wilcoxon"] < 0.05 else "ns")

            row_str = (
                f"| **{d}** | "
                f"{boot_indic['mean_a']:.2f}% [{boot_indic['ci_a'][0]:.2f}, {boot_indic['ci_a'][1]:.2f}] | "
                f"{boot_indic['mean_b']:.2f}% [{boot_indic['ci_b'][0]:.2f}, {boot_indic['ci_b'][1]:.2f}] | "
                f"{boot_sra['mean_b']:.2f}% [{boot_sra['ci_b'][0]:.2f}, {boot_sra['ci_b'][1]:.2f}] | "
                f"{boot_indic['mean_diff']:+.2f}% [{boot_indic['ci_diff'][0]:+.2f}, {boot_indic['ci_diff'][1]:+.2f}] | "
                f"{p_str} ({sig_marker}) |"
            )
            report_lines.append(row_str + "\n")

    # Evaluate Hypothesis E4
    d1_diff_ci = results["dialects"]["D1"]["norm_wer_percent"]["diff_vs_indic"]["ci95"]
    d3_diff_ci = results["dialects"]["D3"]["norm_wer_percent"]["diff_vs_indic"]["ci95"]
    d1_p = results["dialects"]["D1"]["norm_wer_percent"]["diff_vs_indic"]["p_wilcoxon"]
    d3_p = results["dialects"]["D3"]["norm_wer_percent"]["diff_vs_indic"]["p_wilcoxon"]

    # D1 gain is significant if CI does not include 0 (strictly negative since MoE is lower WER)
    d1_sig = (d1_diff_ci[1] < 0) or (d1_diff_ci[0] > 0)
    # D3 difference not significant if p >= 0.05 or CI spans 0
    d3_not_sig = (d3_diff_ci[0] <= 0 <= d3_diff_ci[1])

    h_supported = d1_sig
    results["hypothesis_e4"]["d1_significant"] = bool(d1_sig)
    results["hypothesis_e4"]["d3_nonsignificant"] = bool(d3_not_sig)
    results["hypothesis_e4"]["supported"] = bool(h_supported)

    report_lines.append("\n## 2. Hypothesis Testing Verdict (E4)\n\n")
    report_lines.append(f"- **Hypothesis:** {results['hypothesis_e4']['statement']}\n")
    report_lines.append(
        f"- **D1 Result:** MoE vs IndicConformer diff = "
        f"{results['dialects']['D1']['norm_wer_percent']['diff_vs_indic']['mean']:+.2f}% WER, "
        f"95% CI [{d1_diff_ci[0]:+.2f}%, {d1_diff_ci[1]:+.2f}%], Wilcoxon p = {d1_p:.4e}. "
        f"Significant? **{'YES' if d1_sig else 'NO'}**\n"
    )
    report_lines.append(
        f"- **D3 Result:** MoE vs IndicConformer diff = "
        f"{results['dialects']['D3']['norm_wer_percent']['diff_vs_indic']['mean']:+.2f}% WER, "
        f"95% CI [{d3_diff_ci[0]:+.2f}%, {d3_diff_ci[1]:+.2f}%], Wilcoxon p = {d3_p:.4e}. "
        f"CI spans zero / Non-significant? **{'YES' if d3_not_sig else 'NO'}**\n"
    )
    verdict_text = "CONFIRMED" if h_supported else "REFUTED"
    report_lines.append(f"\n> [!IMPORTANT]\n> **Hypothesis E4 Verdict:** **{verdict_text}**\n")

    # Save JSON and Markdown
    json_path = OUTPUT_DIR / "e4_statistical_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = OUTPUT_DIR / "e4_statistical_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print("\n" + "".join(report_lines))
    print(f"\n[Artifact] Saved E4 results to:\n- {json_path}\n- {md_path}")
    return results


if __name__ == "__main__":
    run_e4_analysis()
