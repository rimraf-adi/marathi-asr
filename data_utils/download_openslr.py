import os
import urllib.request
import zipfile
import json
import csv
from pathlib import Path

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"[{dest_path.name}] Already exists, skipping download.")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded {dest_path.name}")

def main():
    print("[OpenSLR64] Preparing OpenSLR64 Marathi Dataset for Final Alignment...")
    out_dir = Path("data/openslr64_marathi")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    zip_url = "https://openslr.trmal.net/resources/64/mr_in_female.zip"
    tsv_url = "https://openslr.trmal.net/resources/64/line_index.tsv"
    
    zip_path = out_dir / "mr_in_female.zip"
    tsv_path = out_dir / "line_index.tsv"
    
    download_file(zip_url, zip_path)
    download_file(tsv_url, tsv_path)
    
    audio_dir = out_dir / "wavs"
    if not audio_dir.exists():
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(audio_dir)
        print("Extraction complete.")
    else:
        print("Audio directory already extracted.")
        
    # Read TSV and build manifest
    manifest_path = out_dir / "manifest.json"
    manifest_entries = []
    
    print("Building manifest...")
    # The TSV format: FileID \t Transcription
    with open(tsv_path, "r", encoding="utf-8") as f:
        # TSV might not have headers
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                file_id = parts[0]
                text = parts[1]
                
                # audio files in zip are named like mr_f_xxxxx.wav or just file_id.wav
                # Looking at OpenSLR typical structure, it might just be file_id.wav
                wav_file = audio_dir / f"{file_id}.wav"
                
                # Check if it's in a subdirectory
                if not wav_file.exists():
                    # check if the zip extracted a parent folder
                    matches = list(audio_dir.rglob(f"{file_id}.wav"))
                    if matches:
                        wav_file = matches[0]
                    else:
                        continue
                
                # we don't have exact duration without reading it, but we can compute it on the fly in the loader,
                # or compute it now. Let's just set duration to 5.0 default and let loader sort it out if needed,
                # or read it. Reading it is better.
                import soundfile as sf
                try:
                    info = sf.info(str(wav_file))
                    duration = info.duration
                except:
                    duration = 0.0
                
                # Relative path from out_dir
                rel_path = wav_file.relative_to(out_dir).as_posix()
                
                manifest_entries.append({
                    "audio_filepath": rel_path,
                    "text": text,
                    "duration": duration
                })
                
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"[Done] OpenSLR64 manifest created with {len(manifest_entries)} samples.")

if __name__ == "__main__":
    main()
