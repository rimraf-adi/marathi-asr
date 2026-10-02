import os
import sys
import json
import tarfile
import urllib.request
from pathlib import Path

# Kathbath train URL (assuming it's available via Vistaar or AI4Bharat)
# Wait, actually ai4bharat/kathbath is on HuggingFace as a `datasets` stream.
# We can download it using huggingface_hub or datasets library directly.

from datasets import load_dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    print("[Kathbath] Downloading Kathbath Marathi train set from HuggingFace...")
    out_dir = Path("data/vistaar_benchmarks/kathbath_train/marathi")
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # load dataset (train split)
    ds = load_dataset("ai4bharat/kathbath", "marathi", split="train", trust_remote_code=True, token=HF_TOKEN)
    
    print(f"Total samples to download: {len(ds)}")
    
    manifest_entries = []
    
    for i, item in enumerate(tqdm(ds)):
        # item['audio_filepath'] contains 'array' and 'sampling_rate' in HF audio feature
        if "audio_filepath" in item and isinstance(item["audio_filepath"], dict) and "array" in item["audio_filepath"]:
            audio_array = item["audio_filepath"]["array"]
            sr = item["audio_filepath"]["sampling_rate"]
        elif "audio" in item and isinstance(item["audio"], dict) and "array" in item["audio"]:
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
        else:
            raise KeyError(f"Could not find audio array in keys: {list(item.keys())}")
        
        # We can construct a filename
        file_id = item.get("fname", item.get("id", f"sample_{i}"))
        if file_id.endswith(".m4a") or file_id.endswith(".wav"):
            file_name = file_id.rsplit(".", 1)[0] + ".wav"
        else:
            file_name = f"{file_id}.wav"
        save_path = wavs_dir / file_name
        
        # Save audio
        sf.write(str(save_path), audio_array, sr)
        
        # Get duration
        duration = len(audio_array) / sr
        
        manifest_entries.append({
            "audio_filepath": f"kathbath_train/marathi/wavs/{file_name}",
            "duration": duration,
            "text": item.get("transcript", item.get("text", ""))
        })
        
        # Write to manifest progressively to be safe
        if i % 1000 == 0:
            with open(manifest_path, "w", encoding="utf-8") as f:
                for entry in manifest_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    
    # Final write
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"[Done] Saved {len(manifest_entries)} samples to {out_dir}")

if __name__ == "__main__":
    main()
