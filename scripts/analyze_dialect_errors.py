"""Dialect error taxonomy from saved RESPIN predictions (CPU only).

For every model: substitution/deletion/insertion rates per dialect, and recall of dialect-marker words
(words much more frequent in one regional dialect than in the others) plus their most common replacements.
Output: evals/dialect_error_analysis.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from normalizer import normalize  # noqa: E402

RESPIN = Path(r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr")
meta = json.loads((RESPIN / "meta_test_mr.json").read_text(encoding="utf-8"))
dialects = [m["dialect"] for m in meta.values()]
MODELS = ["ours-moe", "ours-dense", "sravaani", "indic-600m"]
preds = {}
for m in MODELS:
    rows = [json.loads(x) for x in (ROOT / "evals" / m / "respin_mr.preds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(dialects), (m, len(rows))
    preds[m] = rows

refs = [normalize(r["ref"]) for r in preds["ours-moe"]]
# ---- dialect marker words from the references
cnt = {d: Counter() for d in ["D1", "D2", "D3", "D4"]}
ntok = Counter()
for r, d in zip(refs, dialects):
    w = r.split(); cnt[d].update(w); ntok[d] += len(w)
markers = {}
for d in ["D1", "D2", "D4"]:
    others = Counter()
    n_oth = 0
    for o in ["D1", "D2", "D3", "D4"]:
        if o != d:
            others.update(cnt[o]); n_oth += ntok[o]
    ms = []
    for w, c in cnt[d].items():
        if c >= 10 and (c / ntok[d]) >= 5 * ((others[w] + 0.5) / n_oth):
            ms.append(w)
    markers[d] = set(ms)
    print(d, "marker words:", len(ms), "covering", sum(cnt[d][w] for w in ms), "of", ntok[d], "tokens; top:", sorted(ms, key=lambda w: -cnt[d][w])[:12])

result = {"markers": {d: {"n_words": len(s), "tokens": sum(cnt[d][w] for w in s), "total_tokens": ntok[d], "top": sorted(s, key=lambda w: -cnt[d][w])[:25]} for d, s in markers.items()}, "models": {}}
for m in MODELS:
    S = {d: [0, 0, 0, 0] for d in ["D1", "D2", "D3", "D4", "ALL"]}  # S, D, I, N
    mk_hit = {d: 0 for d in markers}; mk_tot = {d: 0 for d in markers}
    conf = {d: Counter() for d in markers}
    for r, p, d in zip(refs, preds[m], dialects):
        if not r:
            continue
        h = normalize(p["hyp"])
        o = jiwer.process_words(r, h)
        for k in (d, "ALL"):
            S[k][0] += o.substitutions; S[k][1] += o.deletions; S[k][2] += o.insertions; S[k][3] += len(r.split())
        rw, hw = r.split(), h.split()
        if d in markers:
            for ch in o.alignments[0]:
                for j in range(ch.ref_end_idx - ch.ref_start_idx):
                    w = rw[ch.ref_start_idx + j]
                    if w in markers[d]:
                        mk_tot[d] += 1
                        if ch.type == "equal":
                            mk_hit[d] += 1
                        elif ch.type == "substitute":
                            conf[d][(w, hw[ch.hyp_start_idx + j])] += 1
    result["models"][m] = dict(
        error_rates={d: dict(sub=round(100 * s[0] / s[3], 2), dele=round(100 * s[1] / s[3], 2), ins=round(100 * s[2] / s[3], 2), wer=round(100 * sum(s[:3]) / s[3], 2)) for d, s in S.items()},
        marker_recall={d: round(100 * mk_hit[d] / max(mk_tot[d], 1), 2) for d in markers},
        marker_tokens=mk_tot,
        top_marker_substitutions={d: [[a, b, c] for (a, b), c in conf[d].most_common(10)] for d in markers})
    print(m, "ALL", result["models"][m]["error_rates"]["ALL"], "marker recall", result["models"][m]["marker_recall"], flush=True)
(ROOT / "evals" / "dialect_error_analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
print("done")
