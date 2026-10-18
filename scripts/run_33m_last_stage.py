"""
Orchestrator for 33M Parameter Model Last-Stage Training (Stage 3 Vanilla SFT).
Executes Supervised CTC Calibration across the entire multi-source datasets pool:
  - 100% RESPIN (809,934 utterances across D1-D4 dialects)
  - OpenSLR64 Marathi (1,569 utterances)
  - AI4Bharat Kathbath Clean + Noisy (85,701 utterances)
  - Mozilla Common Voice Marathi (1,205 utterances)
  - Pretraining Streaming Corpus (Vaani + Shrutilipi)

Target: 350,000 steps
Model: 33M Conformer-256 + 3-Dialect Sparse MoE (runs/no-splitformer-moe-only)
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

CONFIG_TELEGRAM = Path(r"D:\marathi-asr\telegram_config.json")
RUN_DIR = Path(r"D:\marathi-asr\runs\no-splitformer-moe-only")
PYTHON_EXE = Path(r"D:\marathi-asr\.venv311\Scripts\python.exe")
STAGE_NAME = "stage3_vanilla_sft"
TOTAL_STEPS = 350000


def send_telegram(message: str):
    if not CONFIG_TELEGRAM.exists():
        return
    try:
        with open(CONFIG_TELEGRAM, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": message, "parse_mode": "HTML"}).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except Exception as e:
        print(f"[Telegram Error] {e}")


def find_latest_checkpoint(ckpt_dir: Path, stage_name: str) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    pts = []
    for p in ckpt_dir.glob(f"{stage_name}_step_*.pt"):
        suffix = p.stem.split("_step_")[-1]
        if suffix.isdigit():
            pts.append((int(suffix), p))
    if pts:
        pts.sort(key=lambda x: x[0], reverse=True)
        return pts[0][1]
    return None


def main():
    sft_ckpt_dir = RUN_DIR / "checkpoints" / STAGE_NAME
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = find_latest_checkpoint(sft_ckpt_dir, STAGE_NAME)

    if latest_ckpt and f"step_{TOTAL_STEPS}" in latest_ckpt.name:
        print(f"[Orchestrator] 33M Model Stage 3 training has reached {TOTAL_STEPS:,} steps! Completed.")
        return

    stage2_ckpt = RUN_DIR / "checkpoints" / "stage2_moe" / "best_model.pt"
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Base MoE checkpoint not found at: {stage2_ckpt}")

    print("\n" + "=" * 80)
    if latest_ckpt:
        print(f"[Orchestrator] Resuming 33M Stage 3 Vanilla SFT from: {latest_ckpt}")
    else:
        print(f"[Orchestrator] Launching 33M Stage 3 Vanilla SFT on ENTIRE Datasets pool for {TOTAL_STEPS:,} steps...")
    print("=" * 80 + "\n")

    header = "🚀 <b>KICKOFF: 33M MODEL STAGE 3 TRAINING (350K TARGET, FULL DATASETS POOL)</b>" if not latest_ckpt else "🚀 <b>RESUMING: 33M MODEL STAGE 3 TRAINING (350K TARGET)</b>"
    send_telegram(f"""{header}
• <b>Model:</b> <code>no-splitformer-moe-only</code> (32.84M Params, 24.84M Trainable)
• <b>Starting State:</b> <code>{'Fresh Start from Stage 2 Best Model' if not latest_ckpt else latest_ckpt.name}</code>
• <b>Datasets Pool (5 Sources Online):</b>
  - 100% RESPIN (809,934 utts)
  - OpenSLR64 Marathi (1,569 utts)
  - Kathbath Clean + Noisy (85,701 utts)
  - Mozilla Common Voice Marathi (1,205 utts)
  - Pretraining Streaming Corpus (Vaani + Shrutilipi)
• <b>Target:</b> {TOTAL_STEPS:,} steps (Batch 64, Grad Accum 2 = 128 effective)
• <b>Learning Rate:</b> 3e-5 → 5e-6 (Warmup 1,000 steps, Cosine Decay)
• <b>Freezing Strategy:</b> Blocks 1-4 Frozen, Blocks 5-12 + CTC Head Unfrozen""")

    cmd = [
        str(PYTHON_EXE), "-u", "final_alignment/train_alignment.py",
        "--run_dir", str(RUN_DIR),
        "--moe_ckpt", str(stage2_ckpt),
        "--checkpoint_dir", str(sft_ckpt_dir),
        "--stage_name", STAGE_NAME,
        "--steps", str(TOTAL_STEPS),
        "--batch_size", "64",
        "--grad_accum_steps", "2",
        "--lr", "3e-5",
        "--min_lr", "5e-6",
        "--warmup_steps", "1000",
        "--log_every", "25",
        "--save_every", "2000",
    ]
    if latest_ckpt:
        cmd.extend(["--resume", str(latest_ckpt)])

    subprocess.run(cmd, cwd=str(Path(r"D:\marathi-asr")), check=True)

    print(f"\n[Orchestrator] 33M Stage 3 Vanilla SFT completed {TOTAL_STEPS:,} steps!")
    send_telegram(f"🏁 <b>33M Model Stage 3 Training Complete!</b> ({TOTAL_STEPS:,} steps reached).")


if __name__ == "__main__":
    main()
