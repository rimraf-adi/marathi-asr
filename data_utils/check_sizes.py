"""Check dataset sizes for all Marathi-related configs from Vaani + Shrutilipi."""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
hf_token = os.getenv("HF_TOKEN")

from huggingface_hub import HfApi

api = HfApi(token=hf_token)

print("=" * 60)
print("SHRUTILIPI — file sizes for Marathi & Konkani")
print("=" * 60)

info_shrutilipi = api.dataset_info("ai4bharat/Shrutilipi", files_metadata=True)
shrutilipi_total = 0
for lang in ["marathi", "konkani"]:
    lang_files = [
        s for s in info_shrutilipi.siblings
        if s.rfilename.startswith(f"{lang}/") and s.size
    ]
    lang_size = sum(s.size for s in lang_files)
    lang_size_gb = lang_size / (1024**3)
    shrutilipi_total += lang_size
    print(f"  {lang:15s}: {len(lang_files):4d} files, {lang_size_gb:6.2f} GB")

print(f"  {'TOTAL':15s}: {shrutilipi_total / (1024**3):6.2f} GB")

print()
print("=" * 60)
print("VAANI — file sizes for Marathi-belt dialects")
print("=" * 60)

info_vaani = api.dataset_info("ARTPARK-IISc/Vaani", files_metadata=True)

# Dialects of Marathi and region in Vaani
vaani_configs = [
    "Marathi",
    "Konkani",
    "Malvani",
    "Khandeshi",
    "Powari",
    "Lambani",
]

vaani_total = 0
for lang in vaani_configs:
    lang_files = [
        s for s in info_vaani.siblings
        if s.rfilename.startswith(f"audio/{lang}/") and s.size
    ]
    lang_size = sum(s.size for s in lang_files)
    lang_size_gb = lang_size / (1024**3)
    vaani_total += lang_size
    print(f"  {lang:15s}: {len(lang_files):4d} files, {lang_size_gb:6.2f} GB")

print(f"  {'TOTAL':15s}: {vaani_total / (1024**3):6.2f} GB")

print()
grand_total = (shrutilipi_total + vaani_total) / (1024**3)
print("=" * 60)
print(f"GRAND TOTAL DOWNLOAD NEEDED : {grand_total:6.2f} GB")
print(f"D: drive currently free     :  69.32 GB")
print(f"Headroom left after download: {69.32 - grand_total:6.2f} GB")
print("=" * 60)
