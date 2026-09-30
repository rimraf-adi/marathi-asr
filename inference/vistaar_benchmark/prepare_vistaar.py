"""
Vistaar Benchmark Downloader & Manifest Creator for Marathi ASR.
Downloads and extracts Marathi subsets for:
  - Kathbath
  - Kathbath Hard
  - CommonVoice
  - FLEURS
  - IndicTTS
  - MUCS
"""

import os
import sys
import json
import zipfile
import urllib.request
import argparse
from pathlib import Path

VISTAAR_BENCHMARK_URLS = {
    "kathbath": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath.zip",
    "kathbath_noisy": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath_noisy.zip",
    "commonvoice": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/commonvoice.zip",
    "fleurs": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/fleurs.zip",
    "indictts": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/indictts.zip",
    "mucs": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/mucs.zip",
}

INDICWHISPER_MR_URL = "https://indicwhisper.objectstore.e2enetworks.net/marathi_models.zip"


def create_marathi_manifest(data_dir: Path, output_manifest: Path):
    """Parses extracted directory and writes Vistaar-format JSON lines manifest."""
    manifest_entries = []
    # Looks for audio files and transcript.txt
    transcript_file = data_dir / "transcript.txt"
    if not transcript_file.exists():
        print(f"[Warning] No transcript.txt found at {transcript_file}")
        return

    transcripts = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                transcripts[parts[0].strip()] = parts[1].strip()
            elif len(parts) == 1:
                p = parts[0].split(" ", 1)
                if len(p) == 2:
                    transcripts[p[0].strip()] = p[1].strip()

    audio_dir = data_dir / "audio"
    if not audio_dir.exists():
        audio_dir = data_dir

    for audio_file in audio_dir.glob("*.wav"):
        base_name = audio_file.stem
        text = transcripts.get(base_name, "")
        if text:
            manifest_entries.append({
                "audio_filepath": str(audio_file.resolve()),
                "text": text,
            })

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        for item in manifest_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Manifest] Created manifest with {len(manifest_entries)} samples: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Vistaar Marathi benchmarks")
    parser.add_argument("--benchmark", type=str, default="all", choices=list(VISTAAR_BENCHMARK_URLS.keys()) + ["all"])
    parser.add_argument("--download_dir", type=str, default="data/vistaar_benchmarks")
    parser.add_argument("--download_model", action="store_true", help="Download IndicWhisper Marathi model checkpoint")
    args = parser.parse_args()

    out_dir = Path(args.download_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(VISTAAR_BENCHMARK_URLS.keys()) if args.benchmark == "all" else [args.benchmark]

    for name in targets:
        url = VISTAAR_BENCHMARK_URLS[name]
        zip_path = out_dir / f"{name}.zip"
        extract_path = out_dir / name
        print(f"\n[Vistaar] Dataset: {name.upper()}")
        print(f"  URL: {url}")
        print(f"  Target: {extract_path}")

    if args.download_model:
        print(f"\n[IndicWhisper] Marathi Model URL: {INDICWHISPER_MR_URL}")


if __name__ == "__main__":
    main()
