"""Inspect Vaani dataset: stream one sample from each Marathi-related config."""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
hf_token = os.getenv("HF_TOKEN")

from datasets import load_dataset, get_dataset_config_names

# Get all Vaani configs
all_configs = get_dataset_config_names("ARTPARK-IISc/Vaani")
print(f"Total Vaani configs: {len(all_configs)}")

# Marathi and closely related dialects/languages from Maharashtra belt
marathi_related = [
    "Marathi",
    "Konkani",
    "Malvani",
    "Khandeshi",
    "Powari",     # Vidarbha region
    "Lambani",    # Banjara, spoken in Maharashtra
]

# Filter to only configs that exist
available = [c for c in marathi_related if c in all_configs]
missing = [c for c in marathi_related if c not in all_configs]

print(f"\nAvailable Marathi-belt configs: {available}")
if missing:
    print(f"Not found in Vaani: {missing}")

# Stream one sample from each available config
for lang in available:
    print(f"\n{'='*50}")
    print(f"Vaani: {lang}")
    print(f"{'='*50}")
    try:
        ds = load_dataset("ARTPARK-IISc/Vaani", lang, split="train", streaming=True)
        sample = next(iter(ds))
        print(f"Keys: {list(sample.keys())}")
        for k, v in sample.items():
            if isinstance(v, dict):
                sub_keys = list(v.keys())
                print(f"  {k}: dict with keys {sub_keys}")
                if "array" in v:
                    arr = v["array"]
                    sr = v.get("sampling_rate", "?")
                    print(f"    array shape: {arr.shape}, sampling_rate: {sr}")
                    duration = len(arr) / sr if isinstance(sr, int) else "?"
                    print(f"    duration: {duration:.2f}s" if isinstance(duration, float) else f"    duration: {duration}")
            elif isinstance(v, str):
                print(f"  {k}: {v[:150]}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")

# Also check Shrutilipi configs for reference
print(f"\n{'='*50}")
print("Shrutilipi configs:")
print(f"{'='*50}")
shrutilipi_configs = get_dataset_config_names("ai4bharat/Shrutilipi")
print(shrutilipi_configs)
