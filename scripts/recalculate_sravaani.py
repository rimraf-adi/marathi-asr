"""
Live Benchmark Re-evaluation for SraVaani-1.0 and IndicConformer-600M on the 2,170 RESPIN test suite.
"""

import os
import sys
import time
import json
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import soundfile as sf
import numpy as np
import jiwer
from tqdm import tqdm

from normalizer import normalize, NORM_VERSION
from data_utils.tokenizer import sanitize_transcript

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {DEVICE}: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def bootstrap_ci(errs: np.ndarray, lens: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(errs)
    if n == 0:
        return [0.0, 0.0]
    wers = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        wers.append(errs[idx].sum() / max(lens[idx].sum(), 1))
    return [
        round(float(np.percentile(wers, 2.5) * 100), 2),
        round(float(np.percentile(wers, 97.5) * 100), 2),
    ]

def load_respin_metadata(test_dir: str = r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr"):
    test_path = Path(test_dir)
    meta_path = test_path / "meta_test_mr.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = []
    for uid, info in meta.items():
        wav_file = test_path / info["wav_path"]
        samples.append({
            "uid": uid,
            "wav_path": str(wav_file),
            "text": info.get("text", "").strip(),
            "dialect": info.get("dialect", "D3"),
            "duration": float(info.get("duration", 0.0)),
        })
    return samples

def compute_metrics(samples, preds):
    by_dialect = {"D1": [], "D2": [], "D3": [], "D4": [], "ALL": []}
    for s, p in zip(samples, preds):
        ref_n = normalize(s["text"])
        hyp_n = normalize(p)
        item = {"ref": ref_n, "hyp": hyp_n}
        by_dialect[s["dialect"]].append(item)
        by_dialect["ALL"].append(item)

    metrics = {}
    for d, items in by_dialect.items():
        refs = [it["ref"] for it in items]
        hyps = [it["hyp"] for it in items]
        keep = [i for i, r in enumerate(refs) if r.strip()]
        r_k = [refs[i] for i in keep]
        h_k = [hyps[i] for i in keep]

        err_list, len_list = [], []
        for r, h in zip(r_k, h_k):
            o = jiwer.process_words(r, h)
            err_list.append(o.substitutions + o.deletions + o.insertions)
            len_list.append(len(r.split()))

        errs = np.array(err_list)
        lens = np.array(len_list)
        wer = round(float(errs.sum() / max(lens.sum(), 1) * 100), 2)
        cer = round(float(jiwer.cer(r_k, h_k) * 100), 2)
        ci = bootstrap_ci(errs, lens) if d == "ALL" else None
        metrics[d] = {"wer": wer, "cer": cer, "n": len(r_k), "ci95": ci}

    return metrics

def run_sravaani_eval(samples, batch_size=32):
    from transformers import AutoModel
    token = os.environ.get("HF_TOKEN")
    print("\n[SraVaani] Loading ARTPARK-IISc/SraVaani-1.0...")
    model = AutoModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0", trust_remote_code=True, token=token).to(DEVICE)
    model.eval()

    wav_paths = [s["wav_path"] for s in samples]
    tot_audio_sec = sum(s["duration"] for s in samples)
    print(f"[SraVaani] Transcribing {len(samples)} utterances (batch_size={batch_size})...")
    t0 = time.perf_counter()
    hyps = model.transcribe(wav_paths, batch_size=batch_size, return_hypotheses=True)
    wall_time = time.perf_counter() - t0

    preds = []
    for h in hyps:
        raw_text = h.text if hasattr(h, "text") else str(h)
        preds.append(sanitize_transcript(raw_text))

    rtfx = tot_audio_sec / max(wall_time, 1e-5)
    metrics = compute_metrics(samples, preds)
    return metrics, rtfx, wall_time

if __name__ == "__main__":
    samples = load_respin_metadata()
    print(f"Loaded {len(samples)} RESPIN test samples.")
    m_sra, rtfx, wall = run_sravaani_eval(samples, batch_size=32)
    print(f"\n=== SraVaani RESPIN Live Results ===")
    print(f"RTFx: {rtfx:.1f}x (Wall time: {wall:.2f}s)")
    for d in ["D1", "D2", "D3", "D4", "ALL"]:
        print(f"  {d}: WER {m_sra[d]['wer']}%, CER {m_sra[d]['cer']}% (N={m_sra[d]['n']})")

    # Save to disk
    out_file = "inference/results/respin/sravaani_fresh_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": m_sra, "rtfx": round(rtfx, 1)}, f, indent=2, ensure_ascii=False)
    print(f"Saved SraVaani live metrics to {out_file}!")
