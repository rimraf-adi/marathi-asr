"""Inference-time ablations of the reported MoE checkpoint (no retraining):
  * bypass the expert delta in a single MoE layer / lower half / upper half
  * drop one expert (router cannot select it)
Greedy CTC, same data/normalizer as eval_offline.py. Output: evals/moe_layer_expert_ablation.json
"""
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from adapters.our_conformer import OurConformer  # noqa: E402
from rerun_paper_evals import MOE_CKPT, greedy, load_items, run_logits, summarize, utt_errors  # noqa: E402

ad = OurConformer(str(MOE_CKPT), "moe")
tok = ad.tokenizer
items = load_items()
layers = [(i, l.ffn2) for i, l in enumerate(ad.model.encoder.layers) if type(l.ffn2).__name__ == "SparseMoELayer"]
print("MoE layers (1-indexed):", [i + 1 for i, _ in layers], flush=True)

orig_fwd = {i: m.forward for i, m in layers}
handles = []


def reset():
    for h in handles:
        h.remove()
    handles.clear()
    for i, m in layers:
        m.forward = orig_fwd[i]


def bypass(idx_list):
    for i, m in layers:
        if i in idx_list:
            f = orig_fwd[i]

            def patched(x, *a, _f=f, **kw):
                kw.pop("dialect_idx", None); kw.pop("use_hard_routing", None)
                return _f(x, dialect_idx=torch.full((x.shape[0],), 3, device=x.device, dtype=torch.long), use_hard_routing=True)

            m.forward = patched


def drop_expert(k):
    for i, m in layers:
        def hook(mod, inp, out, k=k):
            out = out.clone(); out[..., k] = -1e9
            return out
        handles.append(m.router.register_forward_hook(hook))


all_idx = [i for i, _ in layers]
half = len(all_idx) // 2
configs = {"baseline_learned": (lambda: None)}
for i in all_idx:
    configs[f"bypass_layer_{i + 1}"] = (lambda i=i: bypass({i}))
configs[f"bypass_layers_{all_idx[0] + 1}-{all_idx[half - 1] + 1}_lower_half"] = lambda: bypass(set(all_idx[:half]))
configs[f"bypass_layers_{all_idx[half] + 1}-{all_idx[-1] + 1}_upper_half"] = lambda: bypass(set(all_idx[half:]))
for k, name in enumerate(["expert0_malvani", "expert1_ahirani", "expert2_varhadi"]):
    configs[f"drop_{name}"] = (lambda k=k: drop_expert(k))

out = {}
rng = random.Random(0)
for name, setup in configs.items():
    reset(); setup()
    lps = run_logits(ad, items, "learned", rng)
    e, l = utt_errors(items, greedy(tok, lps))
    out[name] = summarize(items, e, l)
    print(name, out[name], flush=True)
    (ROOT / "evals" / "moe_layer_expert_ablation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
reset()
print("done")
