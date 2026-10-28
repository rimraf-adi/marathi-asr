"""Export and push Custom Trained Dense models (130M and 33M) to Hugging Face.

Usage via uv:
    $env:UV_PROJECT_ENVIRONMENT = ".venv311"
    uv run python scripts/export_and_push_dense_models.py
"""

import json
import os
import shutil
from pathlib import Path
import torch
from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")

EXPORT_DIR = Path("exported_models")
RUNS_DIR = Path("runs")
VOCAB_PATH = Path("data_utils/vocab.json")
MODEL_DIR = Path("model")


def create_130m_dense_package():
    target = EXPORT_DIR / "marathi-conformer-130m-dense"
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/3] Preparing 130M Dense model package at {target}...")

    ckpt_path = RUNS_DIR / "scaled-conformer512-dense-3stage" / "checkpoints" / "stage3_final_alignment" / "best_model.pt"
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_weights_path = target / "model.pt"
    torch.save(ckpt["model_state_dict"], model_weights_path)
    print(f"  Saved clean model weights to {model_weights_path} ({model_weights_path.stat().st_size / 1e6:.1f} MB)")

    config = {
        "model_type": "conformer_dense",
        "architectures": ["MarathiConformerDense"],
        "d_model": 512,
        "n_heads": 8,
        "ffn_expansion": 8,
        "num_layers": 12,
        "conv_kernel_size": 31,
        "vocab_size": 105,
        "enable_moe": False,
        "total_params": "130.8M",
        "active_params": "130.8M",
        "sample_rate": 16000,
        "norm_version": "mr-norm-1",
    }
    with open(target / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    shutil.copy2(VOCAB_PATH, target / "vocab.json")

    for f in MODEL_DIR.glob("*.py"):
        shutil.copy2(f, target / f.name)

    readme_content = """---
language:
- mr
license: mit
tags:
- automatic-speech-recognition
- ctc
- conformer
- dense
- marathi
- speech
datasets:
- rimrafadii/respin-iisc
- ai4bharat/Kathbath
- ARTPARK-IISc/Vaani
pipeline_tag: automatic-speech-recognition
---

# Capacity-Matched Dense Marathi Conformer-512 (130M Parameters) 🎙️

A fully dense, capacity-matched Conformer foundation model for Marathi ASR, trained across 300,000 alignment steps to serve as the strict parameter-matched control (130.8M total parameters) against the dialect-conditioned Sparse MoE Conformer.

## Architecture Highlights
- **Base Architecture**: 12-layer Dense Conformer ($d_{\text{model}}=512$, 8 attention heads, FFN expansion factor = 8, kernel size 31).
- **Parameters**: 130,812,425 parameters (100% active per forward pass).
- **Decoder**: CTC greedy and beam decoding over 105 character vocabulary.

## Benchmark Performance (RESPIN Held-out Suite)
- **Malvani (D1)**: 45.29%
- **Ahirani (D2)**: 42.41%
- **Standard (D3)**: 31.33%
- **Varhadi (D4)**: 39.90%
- **Aggregate WER**: **39.82%** (CER: 11.40%)
"""
    with open(target / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    return target


def create_33m_dense_package():
    target = EXPORT_DIR / "marathi-conformer-33m-dense"
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] Preparing 33M Dense model package at {target}...")

    ckpt_path = RUNS_DIR / "no-splitformer-moe-only" / "checkpoints" / "stage1_pretrain" / "best_model.pt"
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_weights_path = target / "model.pt"
    torch.save(ckpt["model_state_dict"], model_weights_path)
    print(f"  Saved clean model weights to {model_weights_path} ({model_weights_path.stat().st_size / 1e6:.1f} MB)")

    config = {
        "model_type": "conformer_dense",
        "architectures": ["MarathiConformerDenseSmall"],
        "d_model": 256,
        "n_heads": 4,
        "ffn_expansion": 4,
        "num_layers": 12,
        "conv_kernel_size": 31,
        "vocab_size": 105,
        "enable_moe": False,
        "total_params": "20.5M",
        "active_params": "20.5M",
        "sample_rate": 16000,
        "norm_version": "mr-norm-1",
    }
    with open(target / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    shutil.copy2(VOCAB_PATH, target / "vocab.json")

    for f in MODEL_DIR.glob("*.py"):
        shutil.copy2(f, target / f.name)

    readme_content = """---
language:
- mr
license: mit
tags:
- automatic-speech-recognition
- ctc
- conformer
- dense
- marathi
- speech
pipeline_tag: automatic-speech-recognition
---

# Dense Marathi Conformer-Small (Dense Trunk Baseline) 🎙️

A lightweight 12-layer dense Conformer backbone trained under self-supervised pretraining (117,000 steps) on Marathi speech.

## Architecture Highlights
- **Base Architecture**: 12-layer Dense Conformer ($d_{\text{model}}=256$, 4 attention heads, kernel size 31).
- **Parameters**: 20.5M parameters (100% active).
- **RESPIN Aggregate WER**: 46.22%
"""
    with open(target / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    return target


def push_to_hub(pkg_dir: Path, repo_id: str):
    api = HfApi(token=TOKEN)
    print(f"\nUploading {pkg_dir.name} to Hugging Face Model Hub: {repo_id}...")
    api.upload_folder(
        folder_path=str(pkg_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {pkg_dir.name} weights, config, and model card",
    )
    print(f"  -> Successfully pushed to: https://huggingface.co/{repo_id}")


def main():
    p130_dense = create_130m_dense_package()
    p33_dense = create_33m_dense_package()

    push_to_hub(p130_dense, "rimrafadii/marathi-conformer-130m-dense")
    push_to_hub(p33_dense, "rimrafadii/marathi-conformer-33m-dense")

    print("\nAll dense models pushed to Hugging Face successfully!")


if __name__ == "__main__":
    main()
