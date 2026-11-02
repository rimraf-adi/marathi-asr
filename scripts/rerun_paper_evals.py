"""Single-protocol rerun of the paper's E2 / E4 / E12 numbers on the 130M MoE and dense checkpoints.

Uses the same checkpoints, text normalizer and RESPIN test set as eval_offline.py, so every number is
comparable to evals/ours-moe and evals/ours-dense. Output: evals/paper_rerun.json
"""
import json
import random
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.our_conformer import OurConformer  # noqa: E402
from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript  # noqa: E402
from normalizer import normalize  # noqa: E402
from eval_offline import resample, SR, RESPIN_DIR  # noqa: E402

MOE_CKPT = ROOT / "runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt"
DIALECT_IDX = {"D1": 0, "D2": 1, "D4": 2, "D3": 3}  # 3 -> no expert selected (trunk only)
BATCH = 16
OUT = ROOT / "evals" / "paper_rerun.json"


def load_items():
    meta = json.loads((RESPIN_DIR / "meta_test_mr.json").read_text(encoding="utf-8"))
    items = []
    for m in meta.values():
        a, sr = sf.read(RESPIN_DIR / m["wav_path"], dtype="float32")
        items.append(dict(audio=resample(a, sr), text=m["text"], dialect=m["dialect"], speaker=m.get("speaker_id", "?")))
    return items


def utt_errors(items, hyps):
    errs, lens = [], []
    for it, h in zip(items, hyps):
        r, hn = normalize(it["text"]), normalize(h)
        if not r:
            errs.append(0); lens.append(0); continue
        o = jiwer.process_words(r, hn)
        errs.append(o.substitutions + o.deletions + o.insertions)
        lens.append(len(r.split()))
    return np.array(errs), np.array(lens)


def summarize(items, errs, lens):
    out = {}
    d = np.array([it["dialect"] for it in items])
    for k in ["D1", "D2", "D3", "D4"]:
        m = d == k
        out[k] = round(float(errs[m].sum() / lens[m].sum() * 100), 2)
    out["ALL"] = round(float(errs.sum() / lens.sum() * 100), 2)
    return out


@torch.no_grad()
def run_logits(ad, items, mode, rng):
    """Returns list of (T_i, V) log-prob arrays, one per utterance, under a routing mode."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["audio"]))
    out = [None] * len(items)
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        aud = [items[i]["audio"] for i in idx]
        lens = [len(a) for a in aud]
        pad = torch.zeros(len(aud), max(lens), device=ad.device)
        for j, a in enumerate(aud):
            pad[j, :len(a)] = torch.from_numpy(a)
        kw = {}
        if mode == "oracle":
            kw = dict(dialect_idx=torch.tensor([DIALECT_IDX[items[i]["dialect"]] for i in idx], device=ad.device), use_hard_routing=True)
        elif mode == "random":
            kw = dict(dialect_idx=torch.tensor([rng.choice([0, 1, 2]) for _ in idx], device=ad.device), use_hard_routing=True)
        elif mode == "trunk_only":
            kw = dict(dialect_idx=torch.full((len(idx),), 3, device=ad.device), use_hard_routing=True)
        lp = ad.model.forward_ctc(pad, chunk_size=None, **kw)["log_probs"]
        T = lp.shape[1]
        for j, i in enumerate(idx):
            keep = max(1, int(round(T * lens[j] / max(lens))))
            out[i] = lp[j, :keep].float().cpu().numpy()
    return out


def greedy(tok, lps):
    return [sanitize_transcript(tok.ctc_decode(lp.argmax(-1).tolist())) for lp in lps]


def speaker_bootstrap(items, e_a, l_a, e_b, l_b, B=10000, seed=0):
    spk = sorted({it["speaker"] for it in items})
    sid = {s: i for i, s in enumerate(spk)}
    S = len(spk)
    d = np.array([it["dialect"] for it in items])
    idx = np.array([sid[it["speaker"]] for it in items])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(S, np.ones(S) / S, size=B).astype(np.float64)  # B x S
    res = {"n_speakers": S}
    for k in ["D1", "D2", "D3", "D4", "ALL"]:
        m = np.ones(len(items), bool) if k == "ALL" else d == k
        def agg(e, l):
            return (np.bincount(idx[m], weights=e[m], minlength=S), np.bincount(idx[m], weights=l[m], minlength=S))
        ea, la = agg(e_a, l_a); eb, lb = agg(e_b, l_b)
        wa = (counts @ ea) / np.maximum(counts @ la, 1) * 100
        wb = (counts @ eb) / np.maximum(counts @ lb, 1) * 100
        delta = wa - wb  # a = MoE, b = dense -> negative = MoE better
        ci = lambda x: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]
        res[k] = dict(moe_ci=ci(wa), dense_ci=ci(wb), delta_mean=round(float(delta.mean()), 2), delta_ci=ci(delta),
                      p=float(max((delta >= 0).mean(), 1.0 / B)))
    return res


def main():
    t0 = time.time()
    items = load_items()
    print(f"loaded {len(items)} utterances, {len({i['speaker'] for i in items})} speakers", flush=True)
    ad = OurConformer(str(MOE_CKPT), "moe")
    tok = ad.tokenizer
    rng = random.Random(42)
    result = {"checkpoint": str(MOE_CKPT.relative_to(ROOT))}

    # ---- E2: routing modes (greedy CTC) ----
    e2, logits_learned = {}, None
    for mode in ["learned", "oracle", "random", "trunk_only"]:
        lps = run_logits(ad, items, mode, rng)
        if mode == "learned":
            logits_learned = lps
        e, l = utt_errors(items, greedy(tok, lps))
        e2[mode] = summarize(items, e, l)
        if mode == "learned":
            e_moe, l_moe = e, l
        print("E2", mode, e2[mode], flush=True)
    result["E2_routing_greedy"] = e2

    # ---- E12: decoders on the learned-routing logits ----
    from decoding.beam_search_decoder import MarathiBeamSearchDecoder, load_marathi_wordlist
    words = load_marathi_wordlist()
    lm = str(ROOT / "lm" / "marathi_5gram.binary")
    e12 = {"greedy": e2["learned"]}
    cfgs = {"lexicon_beam16": (None, 0.0, 0.0), "kenlm_default": (lm, 0.5, 1.5), "kenlm_soft": (lm, 0.15, 1.0)}
    for name, (path, a, b) in cfgs.items():
        dec = MarathiBeamSearchDecoder(tok, unigram_words=words, kenlm_path=path, alpha=a, beta=b)
        hyps = [dec.decode(lp, beam_width=16) for lp in logits_learned]
        e, l = utt_errors(items, hyps)
        e12[name] = summarize(items, e, l)
        print("E12", name, e12[name], flush=True)
    result["E12_decoding_130m_moe"] = e12

    # ---- E4: speaker-clustered paired bootstrap vs the dense preds from evals/ours-dense ----
    dense = [json.loads(x)["hyp"] for x in (ROOT / "evals/ours-dense/respin_mr.preds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dense) == len(items)
    e_d, l_d = utt_errors(items, dense)
    result["dense_greedy"] = summarize(items, e_d, l_d)
    result["E4_speaker_bootstrap"] = speaker_bootstrap(items, e_moe, l_moe, e_d, l_d)
    print("E4", json.dumps(result["E4_speaker_bootstrap"]["ALL"]), flush=True)

    result["wall_secs"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
