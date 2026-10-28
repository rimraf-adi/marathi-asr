import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jiwer
import numpy as np
import yaml
from datasets import Audio, load_dataset

from normalizer import NORM_VERSION, normalize

HF_TOKEN = os.getenv("HF_TOKEN")


def per_utt_errors(refs: List[str], hyps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    errs, lens = [], []
    for r, h in zip(refs, hyps):
        o = jiwer.process_words(r, h)
        errs.append(o.substitutions + o.deletions + o.insertions)
        lens.append(len(r.split()))
    return np.array(errs), np.array(lens)


def bootstrap_ci(errs: np.ndarray, lens: np.ndarray, n_boot: int = 1000, seed: int = 42) -> List[float]:
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


def evaluate_dataset(adapter, ds_cfg, sample_rate, limit=None, batch=16):
    print(f"\nEvaluating dataset: {ds_cfg['id']} ({ds_cfg['path']})...")
    ds = load_dataset(
        ds_cfg["path"],
        ds_cfg.get("config"),
        split=ds_cfg["split"],
        token=HF_TOKEN,
    )
    ds = ds.cast_column(ds_cfg["audio_col"], Audio(sampling_rate=sample_rate))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    refs, hyps, audio_secs, wall = [], [], 0.0, 0.0
    slice_cols = ds_cfg.get("slice_cols", {})
    slice_vals = {k: [] for k in slice_cols}

    # Warm-up call
    first = ds[0][ds_cfg["audio_col"]]["array"]
    adapter.transcribe([first], sample_rate)

    total_len = len(ds)
    for i in range(0, total_len, batch):
        rows = ds[i : i + batch]
        arrays = [a["array"] for a in rows[ds_cfg["audio_col"]]]
        t0 = time.perf_counter()
        out = adapter.transcribe(arrays, sample_rate)
        wall += time.perf_counter() - t0
        audio_secs += sum(len(a) / sample_rate for a in arrays)
        hyps += out
        refs += rows[ds_cfg["text_col"]]
        for k, col in slice_cols.items():
            if col in rows:
                slice_vals[k] += rows[col]
            else:
                slice_vals[k] += ["unknown"] * len(out)
        if (i // batch) % 10 == 0:
            print(f"  Processed {min(i + batch, total_len)}/{total_len} utterances...")

    n_refs = [normalize(r) for r in refs]
    n_hyps = [normalize(h) for h in hyps]
    keep = [i for i, r in enumerate(n_refs) if r.strip()]
    r = [n_refs[i] for i in keep]
    h = [n_hyps[i] for i in keep]

    errs, lens = per_utt_errors(r, h)
    result = {
        "wer": round(float(errs.sum() / max(lens.sum(), 1) * 100), 2),
        "cer": round(float(jiwer.cer(r, h) * 100), 2),
        "n_utts": len(r),
        "hours": round(audio_secs / 3600, 2),
        "wer_ci95": bootstrap_ci(errs, lens),
    }

    slices = {}
    for k, vals in slice_vals.items():
        groups = {}
        for j, idx in enumerate(keep):
            val = vals[idx] if idx < len(vals) else "unknown"
            g = val if val not in (None, "") else "unknown"
            groups.setdefault(g, []).append(j)
        slices[k] = {
            g: {
                "wer": round(float(errs[ix].sum() / max(lens[ix].sum(), 1) * 100), 2),
                "n": len(ix),
            }
            for g, ix in groups.items()
        }

    return result, slices, audio_secs, wall, list(zip(refs, hyps))


def main():
    parser = argparse.ArgumentParser(description="Marathi ASR Offline Benchmark Harness")
    parser.add_argument("--config", default="benchmark.yaml", help="Path to benchmark.yaml")
    parser.add_argument("--model", required=True, help="HF model ID or local directory")
    parser.add_argument("--whisper", action="store_true", help="Set language='marathi' for Whisper")
    parser.add_argument("--out", default="out", help="Output directory for record and predictions")
    parser.add_argument("--limit", type=int, default=None, help="Max utterances per dataset for quick test")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    from adapters.hf_asr import HFAsr

    adapter = HFAsr(args.model, whisper=args.whisper, batch_size=args.batch_size)

    os.makedirs(args.out, exist_ok=True)
    scores, all_slices, tot_audio, tot_wall = {}, {}, 0.0, 0.0

    for ds_cfg in cfg["datasets"]:
        res, slices, a, w, pairs = evaluate_dataset(
            adapter, ds_cfg, cfg["sample_rate"], limit=args.limit, batch=args.batch_size
        )
        scores[ds_cfg["id"]] = res
        all_slices[ds_cfg["id"]] = slices
        tot_audio += a
        tot_wall += w

        # Save utterance predictions for audit
        pred_path = os.path.join(args.out, f"{ds_cfg['id']}.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for ref, hyp in pairs:
                f.write(json.dumps({"ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")

    record = {
        "model": args.model,
        "type": "open",
        "benchmark_version": cfg["version"],
        "norm_version": NORM_VERSION,
        "rtfx": round(tot_audio / tot_wall, 2) if tot_wall else None,
        "hardware": platform.platform(),
        "language_forced": bool(args.whisper),
        "scores": scores,
        "slices": all_slices,
    }

    record_path = os.path.join(args.out, "record.json")
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print("\n=== Benchmark Evaluation Summary ===")
    print(json.dumps({k: v["wer"] for k, v in scores.items()}, indent=2))
    print(f"\nSaved full benchmark record to: {record_path}")


if __name__ == "__main__":
    main()
