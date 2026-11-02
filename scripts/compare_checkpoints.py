"""Evaluate every candidate checkpoint on RESPIN + Kathbath (same harness as eval_offline.py) and rank them."""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_offline as eo  # noqa: E402
from adapters.our_conformer import OurConformer  # noqa: E402

CANDS = [
    "runs/scaled-conformer512-moe-3stage/checkpoints/stage2_moe/best_model.pt",
    "runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt",
    "runs/scaled-conformer512-moe-3stage/checkpoints/stage3_final_alignment/best_model.pt",
    "runs/moe_hard_routing/checkpoints/stage2_moe/best_model.pt",
    "runs/moe_grpo_kathbath/checkpoints/stage4b_moe_grpo/best_model.pt",
    "runs/stage5_openslr_polish/checkpoints/stage5_grpo_polish/best_model.pt",
    "runs/no-splitformer-moe-only/checkpoints/stage3_vanilla_sft/best_model.pt",
    "runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt",
]
DENSE = len(sys.argv) > 1 and sys.argv[1] == "dense"
if DENSE:
    CANDS = [
        "runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt",
        "runs/scaled-conformer512-dense-3stage/checkpoints/stage2_dense/best_model.pt",
        "runs/scaled-conformer512-dense-3stage/checkpoints/stage1_pretrain/best_model.pt",
    ]
REV = len(sys.argv) > 1 and sys.argv[1] == "rev"
if REV:
    CANDS = CANDS[::-1]
OUT = ROOT / "evals" / ("checkpoint_comparison_dense.json" if DENSE else "checkpoint_comparison_rev.json" if REV else "checkpoint_comparison.json")
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for c in CANDS:
    if c in res:
        continue
    try:
        ad = OurConformer(str(ROOT / c), c)
        row = {}
        for ds in ["respin_mr", "kathbath_mr"]:
            r, sl, _ = eo.run_dataset(ad, ds, None, 16)
            row[ds] = dict(wer=r["wer"], cer=r["cer"], ci=r["wer_ci95"], n=r["n_utts"], dialect={k: v["wer"] for k, v in sl["dialect"].items()} if "dialect" in sl else None)
        res[c] = row
        print("RESULT", c, json.dumps({k: (v["wer"], v["cer"]) for k, v in row.items()}), flush=True)
        del ad
    except Exception as e:
        res[c] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("FAIL", c, res[c], flush=True)
        traceback.print_exc()
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("done")
