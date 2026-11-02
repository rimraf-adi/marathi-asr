"""Kathbath train/valid speaker and sentence overlap audit (clean train + noisy train vs. the valid split used in evals)."""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from normalizer import normalize  # noqa: E402

pf = pq.ParquetFile(hf_hub_download("ai4bharat/Kathbath", "marathi/valid-00000-of-00001.parquet", repo_type="dataset"))
t = pf.read(columns=["fname", "text", "speaker_id", "gender"]).to_pylist()
valid_spk = {str(r["speaker_id"]) for r in t}
valid_txt = {normalize(r["text"] or "") for r in t}
print("valid:", len(t), "utts,", len(valid_spk), "speakers; example speaker_id/fname:", t[0]["speaker_id"], t[0]["fname"])


def read_manifest(p):
    rows = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


res = {"valid_utts": len(t), "valid_speakers": len(valid_spk)}
for name, rel in [("train_clean", "data/vistaar_benchmarks/kathbath_train/marathi/manifest.json"),
                  ("train_noisy", "data/vistaar_benchmarks/kathbath_noisy/marathi/manifest.json")]:
    rows = read_manifest(ROOT / rel)
    # speaker id = leading numeric token of the wav name  <utt>-<speaker>-<gender>.wav
    spk = set()
    for r in rows:
        base = os.path.basename(r["audio_filepath"])
        spk.add(base.split("-")[1])
    txt = {normalize(r["text"]) for r in rows}
    sp_overlap = sorted(spk & valid_spk)
    tx_overlap = txt & valid_txt
    res[name] = dict(utts=len(rows), speakers=len(spk), speaker_overlap=len(sp_overlap),
                     speaker_overlap_pct_of_valid=round(100 * len(sp_overlap) / len(valid_spk), 2),
                     sentence_overlap=len(tx_overlap), sentence_overlap_pct_of_valid=round(100 * len(tx_overlap) / len(valid_txt), 2))
    print(name, res[name], flush=True)
all_train_spk = set()
for name, rel in [("a", "data/vistaar_benchmarks/kathbath_train/marathi/manifest.json"), ("b", "data/vistaar_benchmarks/kathbath_noisy/marathi/manifest.json")]:
    for r in read_manifest(ROOT / rel):
        all_train_spk.add(os.path.basename(r["audio_filepath"]).split("-")[1])
res["any_train_speaker_overlap"] = len(all_train_spk & valid_spk)
res["note"] = "speaker id of train files parsed from the second token of the wav filename (<utt>-<speaker>-<gender>); valid uses the speaker_id column"
print(json.dumps(res, indent=1))
(ROOT / "evals" / "kathbath_overlap_audit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
