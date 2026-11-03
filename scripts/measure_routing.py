"""How frame-level is the learned routing? Batch size 1, 60 utterances per dialect from RESPIN test."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\marathi-asr")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from adapters.our_conformer import OurConformer
from rerun_paper_evals import MOE_CKPT, load_items

ad = OurConformer(str(MOE_CKPT), "moe")
layers = [(i + 1, l.ffn2) for i, l in enumerate(ad.model.encoder.layers) if type(l.ffn2).__name__ == "SparseMoELayer"]
items = load_items()
rng = np.random.default_rng(0)
sel = []
for d in ["D1", "D2", "D3", "D4"]:
    idx = [i for i, it in enumerate(items) if it["dialect"] == d]
    sel += list(rng.choice(idx, 60, replace=False))

K = 3
stats = {b: dict(switch=[], purity=[], maxp=[], use={d: np.zeros(K) for d in ["D1", "D2", "D3", "D4"]}) for b, _ in layers}
for i in sel:
    it = items[i]
    a = torch.from_numpy(it["audio"]).float().unsqueeze(0).to(ad.device)
    with torch.no_grad():
        ad.model.forward_ctc(a, chunk_size=None)
    for b, m in layers:
        idx = m.last_top1_idx[0, :, 0].numpy()  # (T,)
        stats[b]["switch"].append(float((idx[1:] != idx[:-1]).mean()))
        stats[b]["purity"].append(float(np.bincount(idx, minlength=K).max() / len(idx)))
        stats[b]["use"][it["dialect"]] += np.bincount(idx, minlength=K)

# Expected switch rate if frames were assigned independently with the layer's overall expert frequencies
out = {}
for b, _ in layers:
    s = stats[b]
    tot = sum(s["use"].values()); p = tot / tot.sum()
    chance_switch = float(1 - (p ** 2).sum())
    out[b] = dict(
        switch_rate=round(float(np.mean(s["switch"])), 3), chance_switch=round(chance_switch, 3),
        majority_share=round(float(np.mean(s["purity"])), 3),
        usage_pct={d: [round(100 * x / v.sum(), 1) for x in v] for d, v in s["use"].items()})
    print(b, json.dumps({k: out[b][k] for k in ("switch_rate", "chance_switch", "majority_share")}),
          "D1:", out[b]["usage_pct"]["D1"], "D2:", out[b]["usage_pct"]["D2"], "D3:", out[b]["usage_pct"]["D3"], "D4:", out[b]["usage_pct"]["D4"], flush=True)
Path(r"D:\marathi-asr\evals\routing_granularity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
