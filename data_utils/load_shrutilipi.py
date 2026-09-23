"""
Script to load and inspect Marathi and Konkani subsets of ai4bharat/Shrutilipi.
"""

import os
import sys
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

# Set standard output encoding for proper display of Marathi/Konkani Devanagari script
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables (.env)
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)


def load_subset(language: str, streaming: bool = True):
    """
    Load Shrutilipi dataset for a specific language.

    Args:
        language: 'marathi' or 'konkani'
        streaming: If True, streams data on-the-fly without downloading entire 38GB+ dataset.
                   If False, downloads all shards to cache directory.
    """
    print(f"\n==========================================")
    print(f"Loading '{language}' (streaming={streaming})...")
    print(f"==========================================")

    dataset = load_dataset(
        "ai4bharat/Shrutilipi",
        language,
        split="train",
        streaming=streaming,
    )
    return dataset


def inspect_samples(dataset, num_samples: int = 3):
    """Iterate and print first few samples."""
    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break

        audio = sample["audio_filepath"]
        print(f"\n--- Sample #{idx + 1} ---")
        print(f"File Path     : {audio['path']}")
        print(f"Sampling Rate : {audio['sampling_rate']} Hz")
        print(f"Audio Shape   : {audio['array'].shape}")
        print(f"Duration      : {sample['duration']:.2f} seconds")
        print(f"Language Code : {sample['lang']}")
        print(f"Transcription : {sample['text']}")


if __name__ == "__main__":
    # 1. Inspect Marathi (streaming mode)
    marathi_ds = load_subset("marathi", streaming=True)
    inspect_samples(marathi_ds, num_samples=2)

    # 2. Inspect Konkani (streaming mode)
    konkani_ds = load_subset("konkani", streaming=True)
    inspect_samples(konkani_ds, num_samples=2)
