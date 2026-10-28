"""Fresh offline evaluation for the Marathi leaderboard.

One model x several datasets per call. Nothing is read from earlier results: each dataset result is
recomputed unless --resume is passed, and a model record is only assembled from result files that
carry this run's checkpoint hash.

    python eval_offline.py --model ours-moe --datasets respin_mr kathbath_mr indicvoices_mr fleurs_mr
    python eval_offline.py --model ours-dense --limit 40          # smoke test
Needs HF_TOKEN in the environment for gated data and models.
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalizer import NORM_VERSION, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent
SR = 16000
RESPIN_DIR = Path(r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr")

MODELS = {
    "ours-moe": dict(
        display="Marathi Conformer-512 MoE (ours)", kind="ours", params_total_m=130.8, params_active_m=98.9,
        ckpt="runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt",
        url="https://huggingface.co/rimrafadii/marathi-conformer-130m-moe", license="MIT",
        trained_on=["respin_mr", "kathbath_mr"]),
    "ours-dense": dict(
        display="Marathi Conformer-512 Dense (ours)", kind="ours", params_total_m=130.8, params_active_m=130.8,
        ckpt="runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt",
        url="https://huggingface.co/rimrafadii/marathi-conformer-130m-dense", license="MIT",
        trained_on=["respin_mr", "kathbath_mr"]),
    "sravaani": dict(
        display="SraVaani-1.0", kind="sravaani", params_total_m=100, params_active_m=100,
        url="https://huggingface.co/ARTPARK-IISc/SraVaani-1.0", license="see model card", trained_on=None),
    "indic-600m": dict(
        display="IndicConformer-600M", kind="indic", params_total_m=600, params_active_m=600,
        url="https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual", license="see model card", trained_on=None),
}


def resample(a, sr):
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, SR).numpy()
    return a.astype(np.float32)


def hub(repo, fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset", token=os.environ.get("HF_TOKEN"))


def load_respin(limit):
    meta = json.loads((RESPIN_DIR / "meta_test_mr.json").read_text(encoding="utf-8"))
    items = list(meta.values())[:limit] if limit else list(meta.values())
    for m in items:
        a, sr = sf.read(RESPIN_DIR / m["wav_path"], dtype="float32")
        yield resample(a, sr), m["text"], {"dialect": {"D1": "Malvani (D1)", "D2": "Ahirani (D2)", "D3": "Standard (D3)", "D4": "Varhadi (D4)"}.get(m["dialect"], m["dialect"])}


def load_parquet(repo, text_col, slice_cols, limit):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(hub(repo, "marathi/valid-00000-of-00001.parquet"))
    n = 0
    for batch in pf.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            a, sr = sf.read(io.BytesIO(r["audio_filepath"]["bytes"]), dtype="float32")
            yield resample(a, sr), r[text_col] or "", {k: r.get(c) for k, c in slice_cols.items()}
            n += 1
            if limit and n >= limit:
                return


def load_fleurs(limit):
    import csv
    tsv = hub("google/fleurs", "data/mr_in/test.tsv")
    rows = {}
    for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t"):
        rows[r[1]] = (r[2], r[-1] if r[-1] in ("MALE", "FEMALE", "OTHER") else "unknown")  # file_name -> (raw_transcription, gender)
    n = 0
    with tarfile.open(hub("google/fleurs", "data/mr_in/audio/test.tar.gz")) as tf:
        for mem in tf:
            name = os.path.basename(mem.name)
            if name not in rows:
                continue
            a, sr = sf.read(io.BytesIO(tf.extractfile(mem).read()), dtype="float32")
            yield resample(a, sr), rows[name][0], {"gender": rows[name][1]}
            n += 1
            if limit and n >= limit:
                return


LOADERS = {
    "respin_mr": lambda l: load_respin(l),
    "kathbath_mr": lambda l: load_parquet("ai4bharat/Kathbath", "text", {"gender": "gender", "district": "district"}, l),
    "indicvoices_mr": lambda l: load_parquet("ai4bharat/indicvoices", "text", {"gender": "gender", "district": "district"}, l),
    "fleurs_mr": lambda l: load_fleurs(l),
}


def build_adapter(key, spec):
    if spec["kind"] == "ours":
        from adapters.our_conformer import OurConformer
        return OurConformer(str(ROOT / spec["ckpt"]), spec["display"])
    if spec["kind"] == "sravaani":
        from adapters.baselines import SraVaaniBaseline
        return SraVaaniBaseline()
    from adapters.baselines import IndicConformerBaseline
    return IndicConformerBaseline()


def bootstrap(errs, lens, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    w = [errs[i].sum() / max(lens[i].sum(), 1) for i in (rng.integers(0, len(errs), len(errs)) for _ in range(n))]
    return [round(float(np.percentile(w, 2.5) * 100), 2), round(float(np.percentile(w, 97.5) * 100), 2)]


def run_dataset(adapter, ds, limit, batch):
    items = list(LOADERS[ds](limit))
    print(f"  loaded {len(items)} utterances", flush=True)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    hyps = [None] * len(items)
    adapter.transcribe([items[order[0]][0]], SR)  # warm-up, not timed
    wall = 0.0
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = adapter.transcribe([items[i][0] for i in idx], SR)
        except Exception as e:  # a failed batch counts as full deletions, not as skipped
            print(f"  batch failed ({type(e).__name__}: {str(e)[:80]}), scoring as empty", flush=True)
            out = [""] * len(idx)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall += time.perf_counter() - t0
        for i, h in zip(idx, out):
            hyps[i] = h
        if (s // batch) % 20 == 0:
            print(f"  {min(s + batch, len(order))}/{len(order)}", flush=True)
    secs = sum(len(a) for a, _, _ in items) / SR
    refs = [normalize(t) for _, t, _ in items]
    hn = [normalize(h) for h in hyps]
    keep = [i for i, r in enumerate(refs) if r]
    r = [refs[i] for i in keep]
    h = [hn[i] for i in keep]
    errs, lens = [], []
    for a, b in zip(r, h):
        o = jiwer.process_words(a, b)
        errs.append(o.substitutions + o.deletions + o.insertions)
        lens.append(len(a.split()))
    errs, lens = np.array(errs), np.array(lens)
    res = dict(wer=round(float(errs.sum() / lens.sum() * 100), 2), cer=round(float(jiwer.cer(r, h) * 100), 2),
               n_utts=len(r), hours=round(secs / 3600, 3), wer_ci95=bootstrap(errs, lens),
               audio_secs=secs, wall_secs=wall)
    slices = {}
    for k in items[0][2]:
        g = {}
        for j, i in enumerate(keep):
            g.setdefault(str(items[i][2][k] or "unknown"), []).append(j)
        slices[k] = {n: dict(wer=round(float(errs[ix].sum() / max(lens[ix].sum(), 1) * 100), 2), n=len(ix)) for n, ix in g.items()}
    preds = [dict(ref=items[i][1], hyp=hyps[i]) for i in range(len(items))]
    return res, slices, preds


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true", help="reuse dataset results from THIS checkpoint hash only")
    args = ap.parse_args()
    spec = MODELS[args.model]
    work = ROOT / ("evals_smoke" if args.limit else "evals") / args.model
    work.mkdir(parents=True, exist_ok=True)

    ck = None
    if "ckpt" in spec:
        p = ROOT / spec["ckpt"]
        print(f"hashing {p.name} ...", flush=True)
        ck = dict(path=spec["ckpt"].replace("\\", "/"), sha256=sha256(p), size_bytes=p.stat().st_size,
                  mtime=datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))
    else:
        from huggingface_hub import HfApi
        repo = spec["url"].split("huggingface.co/")[1]
        ck = dict(path=f"hf://{repo}", sha256=HfApi(token=os.environ.get("HF_TOKEN")).model_info(repo).sha)  # hub commit
    print("checkpoint:", ck, flush=True)

    adapter = build_adapter(args.model, spec)
    for ds in args.datasets:
        f = work / f"{ds}.json"
        if args.resume and f.exists() and json.loads(f.read_text(encoding="utf-8")).get("checkpoint_sha256") == ck["sha256"]:
            print(f"[{ds}] reuse (same checkpoint hash)")
            continue
        print(f"[{ds}]", flush=True)
        res, slices, preds = run_dataset(adapter, ds, args.limit, args.batch_size)
        print(f"[{ds}] WER {res['wer']}  CER {res['cer']}  n={res['n_utts']}", flush=True)
        f.write_text(json.dumps(dict(checkpoint_sha256=ck["sha256"], result=res, slices=slices), ensure_ascii=False), encoding="utf-8")
        (work / f"{ds}.preds.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds), encoding="utf-8")

    scores, slices, a, w = {}, {}, 0.0, 0.0
    for f in sorted(work.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("checkpoint_sha256") != ck["sha256"]:
            continue
        r = dict(d["result"])
        a += r.pop("audio_secs")
        w += r.pop("wall_secs")
        scores[f.stem] = dict(r, split="valid" if f.stem in ("kathbath_mr", "indicvoices_mr") else "test")
        slices[f.stem] = d["slices"]
    rec = dict(model=spec["url"].split("huggingface.co/")[1], display=spec["display"], url=spec["url"], type="open",
               params_total_m=spec["params_total_m"], params_active_m=spec["params_active_m"], license=spec["license"],
               checkpoint=ck, norm_version=NORM_VERSION, decoder="greedy CTC, no external LM" if spec["kind"] == "ours" else "model default",
               language_forced=spec["kind"] != "ours", eval_date=datetime.date.today().isoformat(),
               hardware=f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} / {platform.system()}",
               rtfx=round(a / w, 1) if w else None, trained_on=spec["trained_on"], scores=scores, slices={k: v for k, v in slices.items() if v})
    out = (work / "record.json") if args.limit else ROOT / "spaces/results" / f"{args.model}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
