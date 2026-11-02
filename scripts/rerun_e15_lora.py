"""E15 rerun: oracle LoRA + cross-adapter swap matrix on the same dense checkpoint/normalizer as evals/ours-dense."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from adapters.our_conformer import OurConformer
from model.lora import inject_dialect_lora, set_active_lora_adapter, load_lora_state_dict
from data_utils.tokenizer import sanitize_transcript
from rerun_paper_evals import load_items, utt_errors, summarize, BATCH

ad = OurConformer(str(ROOT / "runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt"), "dense")
ck = torch.load(ROOT / "runs/dialect_lora_adapters/best_lora_adapters.pt", map_location="cpu", weights_only=False)
cfg = ck["lora_config"]; print("lora_config", cfg)
inject_dialect_lora(ad.model, num_dialects=3, r=cfg.get("r", 16), lora_alpha=cfg.get("lora_alpha", 32.0), lora_dropout=0.0, target_layers="all")
load_lora_state_dict(ad.model, ck["lora_state_dict"]); ad.model.to(ad.device).eval()
items = load_items()
A = {"D1": 0, "D2": 1, "D4": 2, "D3": None}

@torch.no_grad()
def run(sel):  # sel(item)->adapter idx or None
    order = sorted(range(len(items)), key=lambda i: len(items[i]["audio"])); hyps = [None]*len(items)
    # group by adapter so each batch uses one adapter
    groups = {}
    for i in order: groups.setdefault(sel(items[i]), []).append(i)
    for a, ids in groups.items():
        set_active_lora_adapter(ad.model, a)
        for s in range(0, len(ids), BATCH):
            idx = ids[s:s+BATCH]; out = ad.transcribe([items[i]["audio"] for i in idx])
            for i, h in zip(idx, out): hyps[i] = h
    e, l = utt_errors(items, hyps); return summarize(items, e, l)

res = {"no_adapter": run(lambda it: None), "oracle_lora": run(lambda it: A[it["dialect"]])}
for a in range(3):
    res[f"all_adapter_{a}"] = run(lambda it, a=a: a)
print(json.dumps(res, indent=1))
(ROOT / "evals" / "e15_rerun.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
