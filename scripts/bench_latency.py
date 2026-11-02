"""Latency / throughput / peak-memory benchmark of the reported MoE and dense checkpoints (fp32, RTX A5000).
Run on an otherwise idle GPU. Output: evals/latency_benchmark.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adapters.our_conformer import OurConformer  # noqa: E402
from rerun_paper_evals import MOE_CKPT, load_items  # noqa: E402

DENSE_CKPT = ROOT / "runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt"
items = load_items()
rng = np.random.default_rng(0)
sub = [items[i] for i in rng.choice(len(items), 300, replace=False)]
aud = [s["audio"] for s in sub]
secs = sum(len(a) for a in aud) / 16000
out = {"gpu": torch.cuda.get_device_name(0), "n_utts": len(aud), "audio_secs": round(secs, 1), "precision": "fp32"}


def sync():
    torch.cuda.synchronize()


for name, ck in [("moe", MOE_CKPT), ("dense", DENSE_CKPT)]:
    ad = OurConformer(str(ck), name)
    params = sum(p.numel() for p in ad.model.parameters()) / 1e6
    for a in aud[:5]:
        ad.transcribe([a])
    sync(); torch.cuda.reset_peak_memory_stats()
    lat = []
    for a in aud:  # batch 1
        sync(); t = time.perf_counter(); ad.transcribe([a]); sync(); lat.append(time.perf_counter() - t)
    peak1 = torch.cuda.max_memory_allocated() / 2**20
    lat = np.array(lat); durs = np.array([len(a) / 16000 for a in aud])
    order = sorted(range(len(aud)), key=lambda i: len(aud[i]))
    torch.cuda.reset_peak_memory_stats(); sync(); t = time.perf_counter()
    for s in range(0, len(order), 16):
        ad.transcribe([aud[i] for i in order[s:s + 16]])
    sync(); wall16 = time.perf_counter() - t
    peak16 = torch.cuda.max_memory_allocated() / 2**20
    out[name] = dict(params_m=round(params, 1),
                     batch1_latency_ms_p50=round(float(np.percentile(lat, 50) * 1000), 1),
                     batch1_latency_ms_p95=round(float(np.percentile(lat, 95) * 1000), 1),
                     batch1_rtf=round(float(lat.sum() / durs.sum()), 4),
                     batch16_rtf=round(wall16 / secs, 4), batch16_rtfx=round(secs / wall16, 1),
                     peak_vram_mb_batch1=round(peak1), peak_vram_mb_batch16=round(peak16))
    print(name, out[name], flush=True)
    del ad; torch.cuda.empty_cache()
(ROOT / "evals" / "latency_benchmark.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("done")
