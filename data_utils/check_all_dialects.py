"""Check Vaani configs to identify any additional Maharashtra/Marathi-related configs."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
hf_token = os.getenv("HF_TOKEN")

from datasets import get_dataset_config_names
from huggingface_hub import HfApi

api = HfApi(token=hf_token)
all_configs = get_dataset_config_names("ARTPARK-IISc/Vaani")

# List of known Maharashtra / Marathi dialects and tribal languages:
# Ahirani (often under Khandeshi), Varhadi, Dangi, Konkani, Malvani, Powari,
# Gondi, Bhili, Halbi, Kolami, Korku, Lambani, etc.
candidate_dialects = [
    "Marathi", "Konkani", "Malvani", "Khandeshi", "Powari", "Lambani",
    "Gondi", "Bhili", "Halbi", "Korku", "Kolami", "Dangi", "Ahirani", "Varhadi",
    "Bhatri", "Dorli"
]

print("Matching candidates in Vaani 106 configs:")
matched = [c for c in candidate_dialects if c in all_configs]
print(matched)

info = api.dataset_info("ARTPARK-IISc/Vaani", files_metadata=True)

print("\n--- Detailed File Sizes in Vaani ---")
total_size = 0
for cfg in matched:
    files = [s for s in info.siblings if s.rfilename.startswith(f"audio/{cfg}/") and s.size]
    cfg_size = sum(s.size for s in files)
    total_size += cfg_size
    print(f"{cfg:15s}: {len(files):3d} files | {cfg_size / (1024**3):6.2f} GB")

print(f"\nTotal matched Vaani size: {total_size / (1024**3):.2f} GB")
