"""Offline evaluation & Space deployment publisher for Marathi ASR Leaderboard.

Usage via uv:
    $env:UV_PROJECT_ENVIRONMENT = ".venv311"
    uv run python spaces/publish_to_space.py
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("RESULTS_REPO", "rimrafadii/marathi-asr-results")
SPACE_REPO = os.getenv("SPACE_REPO", "rimrafadii/asr-leaderboard")

SPACES_DIR = Path(__file__).resolve().parent


def publish():
    print(f"Connecting to Hugging Face Hub using user token...")
    api = HfApi(token=TOKEN)

    who = api.whoami()
    print(f"Authenticated as: {who.get('name')}")

    results_file = SPACES_DIR / "results.jsonl"
    app_file = SPACES_DIR / "app.py"
    req_file = SPACES_DIR / "requirements.txt"
    readme_file = SPACES_DIR / "README.md"

    # 1. Upload results to dataset repo
    if results_file.exists():
        print(f"\n[1/2] Uploading {results_file.name} to dataset repo '{DATASET_REPO}'...")
        api.upload_file(
            path_or_fileobj=str(results_file),
            path_in_repo="results.jsonl",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message="Update Marathi ASR benchmark results from offline evals",
        )
        print("  -> results.jsonl successfully published to dataset repo!")
    else:
        print(f"  Warning: {results_file} not found!")

    # 2. Upload Space files to HF Space
    print(f"\n[2/2] Publishing Space application to '{SPACE_REPO}'...")
    for f in [app_file, req_file, readme_file, results_file]:
        if f.exists():
            print(f"  Uploading {f.name}...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"Update {f.name} for Marathi ASR Leaderboard Space",
            )
    print("\nAll Space files successfully uploaded! Your Space is live at:")
    print(f"  https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    publish()
