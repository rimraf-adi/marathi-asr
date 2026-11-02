"""Evaluate one ablation checkpoint (RESPIN test + Kathbath valid, greedy CTC, same harness as eval_offline.py).

    python scripts/eval_variant.py --name base_s0 --ckpt runs/ablations/base_s0/checkpoints/stage2_moe/best_model.pt \
        --num_experts 3 --moe_layers 4,5,6,7,8,9,10,11
Writes evals/ablations/<name>.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_offline as eo  # noqa: E402
from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript  # noqa: E402
from moe.upcycling import load_moe_model  # noqa: E402


class VariantAdapter:
    def __init__(self, ckpt, num_experts, moe_layers):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_moe_model(ckpt, device=self.device, num_experts=num_experts, moe_layers=moe_layers).eval()
        self.tokenizer = MarathiTokenizer(str(ROOT / "data_utils/vocab.json"))

    @torch.no_grad()
    def transcribe(self, audio: List[np.ndarray], sample_rate: int = 16000) -> List[str]:
        lens = [len(a) for a in audio]
        pad = torch.zeros(len(audio), max(lens), dtype=torch.float32, device=self.device)
        for i, a in enumerate(audio):
            pad[i, : len(a)] = torch.from_numpy(np.asarray(a, dtype=np.float32))
        lp = self.model.forward_ctc(pad, chunk_size=None)["log_probs"]
        T = lp.shape[1]
        out = []
        for b in range(len(audio)):
            keep = max(1, int(round(T * lens[b] / max(lens))))
            out.append(sanitize_transcript(self.tokenizer.ctc_decode(lp[b, :keep].argmax(-1).cpu().tolist())))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_experts", type=int, default=3)
    ap.add_argument("--moe_layers", default="4,5,6,7,8,9,10,11")
    ap.add_argument("--datasets", nargs="+", default=["respin_mr", "kathbath_mr"])
    a = ap.parse_args()
    layers = [int(x) for x in a.moe_layers.split(",")]
    ad = VariantAdapter(str(ROOT / a.ckpt), a.num_experts, layers)
    out = {"name": a.name, "ckpt": a.ckpt, "num_experts": a.num_experts, "moe_blocks_1indexed": [i + 1 for i in layers]}
    for ds in a.datasets:
        r, sl, _ = eo.run_dataset(ad, ds, None, 16)
        out[ds] = dict(wer=r["wer"], cer=r["cer"], wer_ci95=r["wer_ci95"], n=r["n_utts"],
                       dialect={k.split("(")[-1].rstrip(")"): v["wer"] for k, v in sl.get("dialect", {}).items()})
        print(a.name, ds, out[ds]["wer"], flush=True)
    d = ROOT / "evals" / "ablations"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
