"""Train/test leakage audit for every evaluation set against every training text source.

Training text pool (Stage 3 pool): RESPIN train + calibration (moe/respin_split_cache.json), Kathbath clean+noisy train
manifests, OpenSLR64 manifest, Common Voice Marathi manifest. Evaluation sets: RESPIN test, Kathbath valid,
IndicVoices valid, FLEURS test. Reports (a) speaker overlap where speaker ids exist, (b) exact normalized-sentence overlap.
Output: evals/leakage_audit.json
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import os  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from normalizer import normalize  # noqa: E402

RESPIN_DIR = Path(r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr")


def manifest_texts(p):
    out = []
    txt = Path(p).read_text(encoding="utf-8").strip()
    try:
        d = json.loads(txt)
        rows = d if isinstance(d, list) else list(d.values())
    except json.JSONDecodeError:
        rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
    for r in rows:
        out.append(r.get("text") or r.get("sentence") or "")
    return out


# ---- training pools
cache = json.loads((ROOT / "moe/respin_split_cache.json").read_text(encoding="utf-8"))
train = {
    "respin_train+calibration": [r["text"] for k in cache for r in cache[k]],
    "kathbath_train_clean": manifest_texts(ROOT / "data/vistaar_benchmarks/kathbath_train/marathi/manifest.json"),
    "kathbath_train_noisy": manifest_texts(ROOT / "data/vistaar_benchmarks/kathbath_noisy/marathi/manifest.json"),
    "openslr64": manifest_texts(ROOT / "data/openslr64_marathi/manifest.json"),
    "commonvoice_mr": manifest_texts(ROOT / "data/vistaar_benchmarks/commonvoice/marathi/manifest.json"),
}
train_sets = {k: {normalize(t) for t in v if t} for k, v in train.items()}
all_train = set().union(*train_sets.values())
res = {"training_text_sources": {k: {"utts": len(train[k]), "unique_sentences": len(train_sets[k])} for k in train}}
print(res, flush=True)

# ---- eval sets (texts, speakers)
ev = {}
meta = json.loads((RESPIN_DIR / "meta_test_mr.json").read_text(encoding="utf-8"))
ev["respin_test"] = dict(texts=[m["text"] for m in meta.values()], speakers={str(m["speaker_id"]) for m in meta.values()})
tr_spk = {str(r["speaker_id"]) for k in cache for r in cache[k]}
for name, repo, f in [("kathbath_valid", "ai4bharat/Kathbath", "marathi/valid-00000-of-00001.parquet"),
                      ("indicvoices_valid", "ai4bharat/indicvoices", "marathi/valid-00000-of-00001.parquet")]:
    pf = pq.ParquetFile(hf_hub_download(repo, f, repo_type="dataset"))
    cols = [c for c in ("text", "speaker_id") if c in pf.schema_arrow.names]
    rows = pf.read(columns=cols).to_pylist()
    ev[name] = dict(texts=[r["text"] or "" for r in rows], speakers={str(r["speaker_id"]) for r in rows} if "speaker_id" in cols else None)
tsv = hf_hub_download("google/fleurs", "data/mr_in/test.tsv", repo_type="dataset")
ev["fleurs_test"] = dict(texts=[r[2] for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t")], speakers=None)

res["eval_sets"] = {}
for name, d in ev.items():
    nt = {normalize(t) for t in d["texts"] if t}
    row = {"utts": len(d["texts"]), "unique_sentences": len(nt),
           "sentence_overlap_with_any_training_text": len(nt & all_train),
           "sentence_overlap_pct": round(100 * len(nt & all_train) / max(len(nt), 1), 2),
           "sentence_overlap_by_source": {k: len(nt & s) for k, s in train_sets.items()}}
    if d["speakers"] is not None:
        row["speakers"] = len(d["speakers"])
    res["eval_sets"][name] = row
# speaker overlap
res["eval_sets"]["respin_test"]["speaker_overlap_with_respin_train+calibration"] = len(ev["respin_test"]["speakers"] & tr_spk)
res["eval_sets"]["respin_test"]["train_calibration_speakers"] = len(tr_spk)
kb = json.loads((ROOT / "evals/kathbath_overlap_audit.json").read_text(encoding="utf-8"))
res["eval_sets"]["kathbath_valid"]["speaker_overlap_with_kathbath_train_clean"] = kb["train_clean"]["speaker_overlap"]
res["eval_sets"]["kathbath_valid"]["speaker_overlap_with_kathbath_train_noisy"] = kb["train_noisy"]["speaker_overlap"]
res["eval_sets"]["kathbath_valid"]["valid_speakers"] = kb["valid_speakers"]
print(json.dumps(res, indent=1, ensure_ascii=False))
(ROOT / "evals" / "leakage_audit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
