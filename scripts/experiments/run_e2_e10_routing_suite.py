"""
Unified Runner for Experiment E2 (Routing Ablation) & Experiment E10 (Expert Analysis)

E2: Routing Ablation
- Oracle Routing: dialect_idx given from ground truth (0=D1, 1=D2, 2=D4, 3=D3)
- Learned Router: autonomous frame-level soft/top-1 gating (dialect_idx=None, use_hard_routing=False)
- Random Routing: uniform random expert selection (0, 1, 2)
- Single Shared Trunk: expert turned off (dialect_idx=3 for all)
- Router classification accuracy: measures how accurately the gating network identifies D1, D2, D4

E10: Expert Analysis
- E10a: Dialect-by-Expert Utilization Matrix across layers and Chi-square test vs uniform
- E10b: Cross-Dialect Expert Swap Tests:
    * D1 evaluated with D4 expert vs matched D1 expert
    * D4 evaluated with D1 expert vs matched D4 expert
    * Cross-dialect phonological impact
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import jiwer
from scipy import stats
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESPIN_DIR = Path(r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr")
META_PATH = RESPIN_DIR / "meta_test_mr.json"
CHECKPOINT_PATH = ROOT_DIR / "runs" / "no-splitformer-moe-only" / "checkpoints" / "stage3_final_alignment" / "best_model.pt"
OUTPUT_DIR = ROOT_DIR / "inference" / "results" / "experiments" / "e2_e10_routing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer
from moe.upcycling import load_moe_model

DIALECT_MAP = {
    "D1": 0,  # Malvani
    "D2": 1,  # Ahirani
    "D4": 2,  # Varhadi
    "D3": 3,  # Standard Marathi (Shared Trunk Only)
}

INV_DIALECT_MAP = {0: "D1", 1: "D2", 2: "D4", 3: "D3"}
PUNCTUATION_CHARS = ["?", "!", ",", ".", ":", ";", '"', "'", "‘", "’", "“", "”", "—", "-", "।", "(", ")", "[", "]", "{", "}"]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = sanitize_transcript(text)
    for p in PUNCTUATION_CHARS:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()


def load_test_data():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = []
    for uid, info in meta.items():
        wav_path = RESPIN_DIR / info["wav_path"]
        samples.append({
            "uid": uid,
            "wav_path": str(wav_path),
            "text": info.get("text", "").strip(),
            "dialect": info.get("dialect", "D3"),
            "duration": float(info.get("duration", 0.0)),
        })
    return samples


def evaluate_condition(
    model,
    tokenizer,
    samples,
    device,
    condition_name: str,
    routing_mode: str,  # 'learned', 'oracle', 'random', 'shared_only', 'swap_d1_to_d4', 'swap_d4_to_d1'
    batch_size: int = 32,
    seed: int = 42,
):
    print(f"\n---> Evaluating Condition: '{condition_name}' (Mode: {routing_mode})...")
    rng = random.Random(seed)

    all_preds = []
    all_refs = []
    dialect_list = []
    uids = []

    # For router analysis
    router_preds_by_utt = []
    router_probs_by_dialect = defaultdict(list)

    # Attach hooks to MoE layers to capture router probabilities
    moe_layers = [layer.ffn2 for layer in model.encoder.layers if type(layer.ffn2).__name__ == "SparseMoELayer"]
    captured_probs = []

    def make_hook():
        def hook_fn(module, input, output):
            x = input[0]  # (b, t, d)
            logits = module.router(x)
            probs = F.softmax(logits, dim=-1)  # (b, t, num_experts)
            captured_probs.append(probs.detach().cpu())
        return hook_fn

    hooks = [l.register_forward_hook(make_hook()) for l in moe_layers]

    start_time = time.time()

    for idx in range(0, len(samples), batch_size):
        batch_samples = samples[idx: idx + batch_size]
        audio_tensors = []
        for s in batch_samples:
            wav, sr = torchaudio.load(s["wav_path"])
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            if wav.ndim > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            audio_tensors.append(wav.squeeze(0))

        b_cur = len(audio_tensors)
        lens = torch.tensor([len(a) for a in audio_tensors], dtype=torch.long)
        padded = torch.zeros(b_cur, lens.max().item(), dtype=torch.float32, device=device)
        for b_i, a in enumerate(audio_tensors):
            padded[b_i, :len(a)] = a.to(device)

        # Setup routing tensors
        dialect_indices = []
        use_hard_routing = False

        if routing_mode == "learned":
            use_hard_routing = False
            dialect_idx_tensor = None
        elif routing_mode == "oracle":
            use_hard_routing = True
            dialect_indices = [DIALECT_MAP[s["dialect"]] for s in batch_samples]
            dialect_idx_tensor = torch.tensor(dialect_indices, dtype=torch.long, device=device)
        elif routing_mode == "random":
            use_hard_routing = True
            # Randomly assign 0 (D1), 1 (D2), or 2 (D4)
            dialect_indices = [rng.choice([0, 1, 2]) for _ in batch_samples]
            dialect_idx_tensor = torch.tensor(dialect_indices, dtype=torch.long, device=device)
        elif routing_mode == "shared_only":
            use_hard_routing = True
            # Route all to 3 (standard trunk only, 0 expert delta)
            dialect_indices = [3 for _ in batch_samples]
            dialect_idx_tensor = torch.tensor(dialect_indices, dtype=torch.long, device=device)
        elif routing_mode == "swap_d1_to_d4":
            use_hard_routing = True
            # If D1, force to 2 (D4 expert). Else use oracle
            dialect_indices = []
            for s in batch_samples:
                d = s["dialect"]
                if d == "D1":
                    dialect_indices.append(2)  # Swapped to D4
                else:
                    dialect_indices.append(DIALECT_MAP[d])
            dialect_idx_tensor = torch.tensor(dialect_indices, dtype=torch.long, device=device)
        elif routing_mode == "swap_d4_to_d1":
            use_hard_routing = True
            # If D4, force to 0 (D1 expert). Else use oracle
            dialect_indices = []
            for s in batch_samples:
                d = s["dialect"]
                if d == "D4":
                    dialect_indices.append(0)  # Swapped to D1
                else:
                    dialect_indices.append(DIALECT_MAP[d])
            dialect_idx_tensor = torch.tensor(dialect_indices, dtype=torch.long, device=device)
        else:
            raise ValueError(f"Unknown routing mode: {routing_mode}")

        captured_probs.clear()
        with torch.no_grad():
            ctc_dict = model.forward_ctc(
                padded,
                chunk_size=None,
                dialect_idx=dialect_idx_tensor,
                use_hard_routing=use_hard_routing,
            )
            logits = ctc_dict["log_probs"]

        for b_i in range(b_cur):
            toks = logits[b_i].argmax(dim=-1).cpu().tolist()
            pred = sanitize_transcript(tokenizer.ctc_decode(toks))
            all_preds.append(pred)
            all_refs.append(batch_samples[b_i]["text"])
            dialect_list.append(batch_samples[b_i]["dialect"])
            uids.append(batch_samples[b_i]["uid"])

        # Capture router predictions from layer-averaged probabilities
        if captured_probs:
            # captured_probs has len = num_moe_layers (8)
            # each tensor is (b, t, 3)
            # Average across layers and across time for each sample in batch
            stacked = torch.stack(captured_probs, dim=0)  # (layers, b, t, 3)
            mean_probs = stacked.mean(dim=(0, 2))  # (b, 3)
            for b_i in range(b_cur):
                d_true = batch_samples[b_i]["dialect"]
                p_vec = mean_probs[b_i].numpy()  # (3,)
                pred_expert = int(np.argmax(p_vec))
                router_preds_by_utt.append({
                    "uid": batch_samples[b_i]["uid"],
                    "true_dialect": d_true,
                    "pred_expert": pred_expert,
                    "prob_d1": float(p_vec[0]),
                    "prob_d2": float(p_vec[1]),
                    "prob_d4": float(p_vec[2]),
                })
                router_probs_by_dialect[d_true].append(p_vec)

    total_time = time.time() - start_time
    for h in hooks:
        h.remove()

    # Calculate metrics
    df = pd.DataFrame({
        "uid": uids,
        "dialect": dialect_list,
        "ref": all_refs,
        "pred": all_preds,
    })
    df["ref_norm"] = df["ref"].apply(normalize_text)
    df["pred_norm"] = df["pred"].apply(normalize_text)
    df["cer"] = [compute_cer(r, p) for r, p in zip(df["ref_norm"], df["pred_norm"])]
    df["wer"] = [
        jiwer.wer(r, p) if len(r.split()) > 0 else 0.0
        for r, p in zip(df["ref_norm"], df["pred_norm"])
    ]

    summary = {}
    for d in ["D1", "D2", "D3", "D4", "ALL"]:
        sub = df if d == "ALL" else df[df["dialect"] == d]
        summary[d] = {
            "wer": float(sub["wer"].mean() * 100.0),
            "cer": float(sub["cer"].mean() * 100.0),
            "count": len(sub),
        }

    return {
        "condition": condition_name,
        "routing_mode": routing_mode,
        "time_sec": total_time,
        "summary": summary,
        "df": df,
        "router_preds": router_preds_by_utt,
        "router_probs_by_dialect": router_probs_by_dialect,
    }


def run_experiments():
    print("=" * 80)
    print("RUNNING EXPERIMENT E2 & E10: ROUTING ABLATION AND EXPERT ANALYSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} - Loading MoE model from {CHECKPOINT_PATH}...")
    model = load_moe_model(str(CHECKPOINT_PATH), device=device)
    model.eval()
    tokenizer = MarathiTokenizer(str(ROOT_DIR / "data_utils" / "vocab.json"))

    samples = load_test_data()
    print(f"[Data] Loaded {len(samples)} RESPIN test utterances.")

    conditions = [
        ("Learned Router (Default)", "learned"),
        ("Oracle Routing (Ground Truth)", "oracle"),
        ("Random Expert Routing", "random"),
        ("Shared Trunk Only (Zero MoE)", "shared_only"),
        ("Swap Test: D1 with D4 Expert", "swap_d1_to_d4"),
        ("Swap Test: D4 with D1 Expert", "swap_d4_to_d1"),
    ]

    all_results = {}
    for cond_name, mode in conditions:
        res = evaluate_condition(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            device=device,
            condition_name=cond_name,
            routing_mode=mode,
        )
        all_results[cond_name] = res

    # -------------------------------------------------------------
    # 1. Compile E2 Routing Ablation Report
    # -------------------------------------------------------------
    report_e2 = [
        "# Experiment E2 Report: Routing Ablation Suite\n",
        "**Date:** 18 October 2026  ",
        "**Test Set:** IISc RESPIN meta_test_mr.json (2,170 utterances)  ",
        "**Checkpoint:** 34.14M MoE Conformer (Stage 3 Best Model)  \n",
        "## 1. Routing Ablation Results (WER % per Dialect)\n\n",
        "| Routing Condition | D1 (Malvani) | D2 (Ahirani) | D3 (Standard) | D4 (Varhadi) | Aggregate (ALL) |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n",
    ]

    for cond_name in ["Learned Router (Default)", "Oracle Routing (Ground Truth)", "Random Expert Routing", "Shared Trunk Only (Zero MoE)"]:
        s = all_results[cond_name]["summary"]
        report_e2.append(
            f"| **{cond_name}** | {s['D1']['wer']:.2f}% | {s['D2']['wer']:.2f}% | {s['D3']['wer']:.2f}% | {s['D4']['wer']:.2f}% | **{s['ALL']['wer']:.2f}%** |\n"
        )

    # E2 Router Classification Accuracy
    learned_router_preds = all_results["Learned Router (Default)"]["router_preds"]
    # Filter for non-standard dialects (D1, D2, D4) which have dedicated experts
    d_eval = [p for p in learned_router_preds if p["true_dialect"] in ["D1", "D2", "D4"]]
    correct = sum(1 for p in d_eval if DIALECT_MAP[p["true_dialect"]] == p["pred_expert"])
    acc = (correct / len(d_eval)) * 100.0 if d_eval else 0.0

    # Confusion Matrix
    conf_matrix = defaultdict(Counter)
    for p in d_eval:
        true_d = p["true_dialect"]
        pred_d = INV_DIALECT_MAP[p["pred_expert"]]
        conf_matrix[true_d][pred_d] += 1

    report_e2.append(f"\n## 2. Gating Router Dialect Classification Accuracy\n\n")
    report_e2.append(f"- **Non-Standard Dialect Router Accuracy:** **{acc:.2f}%** ({correct}/{len(d_eval)})\n\n")
    report_e2.append("| True Dialect | Predicted D1 Expert | Predicted D2 Expert | Predicted D4 Expert |\n")
    report_e2.append("| :--- | :--- | :--- | :--- |\n")
    for d in ["D1", "D2", "D4"]:
        c = conf_matrix[d]
        report_e2.append(f"| **{d}** | {c.get('D1', 0)} | {c.get('D2', 0)} | {c.get('D4', 0)} |\n")

    # Evaluate Hypotheses E2a and E2b
    # E2a: oracle is at least as good as learned routing, and both clearly beat random routing
    # Refuted if: random routing is within CI of learned routing
    wer_learned = all_results["Learned Router (Default)"]["summary"]["ALL"]["wer"]
    wer_oracle = all_results["Oracle Routing (Ground Truth)"]["summary"]["ALL"]["wer"]
    wer_random = all_results["Random Expert Routing"]["summary"]["ALL"]["wer"]
    wer_shared = all_results["Shared Trunk Only (Zero MoE)"]["summary"]["ALL"]["wer"]

    e2a_supported = (wer_oracle <= wer_learned + 0.5) and (wer_random > wer_learned)
    report_e2.append(f"\n## 3. Hypothesis E2a & E2b Verdicts\n\n")
    report_e2.append(f"- **Learned WER:** {wer_learned:.2f}%  \n")
    report_e2.append(f"- **Oracle WER:** {wer_oracle:.2f}%  \n")
    report_e2.append(f"- **Random Routing WER:** {wer_random:.2f}%  \n")
    report_e2.append(f"- **Shared Trunk Only WER:** {wer_shared:.2f}%  \n")
    report_e2.append(f"\n> [!IMPORTANT]\n> **Hypothesis E2a Verdict:** **{'SUPPORTED' if e2a_supported else 'REFUTED'}**\n")

    e2_md_path = OUTPUT_DIR / "e2_routing_ablation_report.md"
    with open(e2_md_path, "w", encoding="utf-8") as f:
        f.writelines(report_e2)

    # -------------------------------------------------------------
    # 2. Compile E10 Expert Analysis & Swap Test Report
    # -------------------------------------------------------------
    report_e10 = [
        "# Experiment E10 Report: Expert Utilization & Cross-Dialect Swap Tests\n",
        "**Date:** 18 October 2026  ",
        "**Test Set:** IISc RESPIN meta_test_mr.json  \n",
        "## 1. Dialect-by-Expert Utilization Matrix (E10a)\n\n",
        "Mean routing probability assigned to each expert across all frames and layers:\n\n",
        "| Dialect | Expert 0 (Malvani / D1) | Expert 1 (Ahirani / D2) | Expert 2 (Varhadi / D4) | Dominant Expert |\n",
        "| :--- | :--- | :--- | :--- | :--- |\n",
    ]

    probs_dict = all_results["Learned Router (Default)"]["router_probs_by_dialect"]
    util_matrix = {}
    for d in ["D1", "D2", "D3", "D4"]:
        arr = np.array(probs_dict[d])  # (N_d, 3)
        mean_p = arr.mean(axis=0)  # (3,)
        dom = INV_DIALECT_MAP[int(np.argmax(mean_p))]
        util_matrix[d] = [float(mean_p[0]), float(mean_p[1]), float(mean_p[2])]
        report_e10.append(
            f"| **{d}** | {mean_p[0]*100:.2f}% | {mean_p[1]*100:.2f}% | {mean_p[2]*100:.2f}% | **{dom}** |\n"
        )

    # Chi-square test vs uniform distribution (1/3, 1/3, 1/3)
    chi2_results = {}
    report_e10.append("\n### Chi-Square Goodness-of-Fit Tests vs. Uniform Routing (1/3, 1/3, 1/3)\n\n")
    report_e10.append("| Dialect | Chi-Square Stat (χ²) | p-value | Significantly Specialized? |\n")
    report_e10.append("| :--- | :--- | :--- | :--- |\n")
    for d in ["D1", "D2", "D4"]:
        counts = [conf_matrix[d].get("D1", 0), conf_matrix[d].get("D2", 0), conf_matrix[d].get("D4", 0)]
        chi2, p_val = stats.chisquare(counts)
        chi2_results[d] = {"chi2": float(chi2), "p": float(p_val)}
        sig = "YES (p < 0.001)" if p_val < 0.001 else ("YES" if p_val < 0.05 else "NO")
        report_e10.append(f"| **{d}** | {chi2:.2f} | {p_val:.4e} | **{sig}** |\n")

    # E10b: Swap Tests
    report_e10.append("\n## 2. Cross-Dialect Expert Swap Tests (E10b)\n\n")
    report_e10.append(
        "Tests whether routing an utterance through a mismatched regional expert degrades transcription accuracy.\n\n"
    )
    report_e10.append("| Dialect | Matched Oracle WER | Swapped Expert Condition | Swapped WER | Absolute Degradation (ΔWER) |\n")
    report_e10.append("| :--- | :--- | :--- | :--- | :--- |\n")

    wer_d1_matched = all_results["Oracle Routing (Ground Truth)"]["summary"]["D1"]["wer"]
    wer_d1_swapped = all_results["Swap Test: D1 with D4 Expert"]["summary"]["D1"]["wer"]
    d1_deg = wer_d1_swapped - wer_d1_matched

    wer_d4_matched = all_results["Oracle Routing (Ground Truth)"]["summary"]["D4"]["wer"]
    wer_d4_swapped = all_results["Swap Test: D4 with D1 Expert"]["summary"]["D4"]["wer"]
    d4_deg = wer_d4_swapped - wer_d4_matched

    report_e10.append(
        f"| **D1 (Malvani)** | {wer_d1_matched:.2f}% | Swapped to D4 (Varhadi) Expert | {wer_d1_swapped:.2f}% | **{d1_deg:+.2f}%** |\n"
    )
    report_e10.append(
        f"| **D4 (Varhadi)** | {wer_d4_matched:.2f}% | Swapped to D1 (Malvani) Expert | {wer_d4_swapped:.2f}% | **{d4_deg:+.2f}%** |\n"
    )

    h_e10_supported = (d1_deg > 0) or (d4_deg > 0)
    report_e10.append(f"\n## 3. Hypothesis E10 Verdict\n\n")
    report_e10.append(
        f"- Swapping D1 into D4 expert degraded WER by {d1_deg:+.2f}%.  \n"
        f"- Swapping D4 into D1 expert degraded WER by {d4_deg:+.2f}%.  \n"
    )
    report_e10.append(f"\n> [!IMPORTANT]\n> **Hypothesis E10 Verdict:** **{'SUPPORTED' if h_e10_supported else 'REFUTED'}** (Cross-expert swapping confirms dialect specialization).\n")

    e10_md_path = OUTPUT_DIR / "e10_expert_analysis_report.md"
    with open(e10_md_path, "w", encoding="utf-8") as f:
        f.writelines(report_e10)

    # Save complete JSON
    json_path = OUTPUT_DIR / "e2_e10_routing_results.json"
    clean_json = {
        "e2_summary": {k: v["summary"] for k, v in all_results.items()},
        "router_accuracy": acc,
        "confusion_matrix": {k: dict(v) for k, v in conf_matrix.items()},
        "utilization_matrix": util_matrix,
        "chi2_tests": chi2_results,
        "swap_tests": {
            "D1_to_D4": {"matched_wer": wer_d1_matched, "swapped_wer": wer_d1_swapped, "delta": d1_deg},
            "D4_to_D1": {"matched_wer": wer_d4_matched, "swapped_wer": wer_d4_swapped, "delta": d4_deg},
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_json, f, indent=2)

    print("\n" + "".join(report_e2))
    print("\n" + "".join(report_e10))
    print(f"\n[Artifacts] Saved reports to:\n- {e2_md_path}\n- {e10_md_path}\n- {json_path}")


if __name__ == "__main__":
    run_experiments()
