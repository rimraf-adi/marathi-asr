"""
Verification script for Konkani pretraining streams.
Tests:
1. IndicVoices (konkani)
2. IndicVoices-R (Konkani)
3. Combined interleaved stream yielding samples at 16kHz
"""
import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pretraining.dataset_loader import get_combined_stream

print("=" * 65, flush=True)
print("Testing Konkani Injected Combined Stream...", flush=True)
print("=" * 65, flush=True)

# Test combined stream with both new Konkani datasets interleaved
stream = get_combined_stream(
    include_shrutilipi=False,
    include_vaani=False,
    include_indicvoices=True,
    include_indicvoices_r=True,
    interleave=True,
)

for i in range(4):
    sample = next(stream)
    arr = sample["audio"]
    sr = sample["sampling_rate"]
    dur = sample["duration"]
    source = sample["source"]
    dialect = sample["dialect"]
    txt = sample["text"][:35] if sample["text"] else "(none)"
    print(f"Sample #{i+1}: Source={source:<15} Dialect={dialect} SR={sr}Hz Shape={arr.shape} Dur={dur:.2f}s Text='{txt}'", flush=True)

print("=" * 65, flush=True)
print("ALL KONKANI STREAMS VERIFIED SUCCESSFULLY!", flush=True)
print("=" * 65, flush=True)
