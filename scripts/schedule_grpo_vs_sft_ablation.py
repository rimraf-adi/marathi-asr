"""
Automated Orchestrator for GRPO vs. Vanilla SFT Ablation Study:
1. Runs Vanilla SFT (Stage 3) on the ENTIRE datasets pool (100% RESPIN 122.6k utts, OpenSLR64, Kathbath Clean/Noisy, Pretrain stream).
2. Runs GRPO Sequence Polish on the ENTIRE datasets pool using the SFT checkpoint as reference/starting model.
3. Automatically triggers 'inference/grpo_vs_sft/benchmark_grpo_vs_sft.py' on the frozen IISc RESPIN test set (2,170 utterances).
4. Delivers Telegram milestone updates at each transition.
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any

CONFIG_TELEGRAM = Path(r"D:\marathi-asr\telegram_config.json")
RUN_DIR = Path(r"D:\marathi-asr\runs\scaled-conformer512-moe-3stage")
PYTHON_EXE = Path(r"D:\marathi-asr\.venv311\Scripts\python.exe")


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


def run_vanilla_sft():
    sft_ckpt_dir = RUN_DIR / "checkpoints" / "stage3_vanilla_sft"
    latest_ckpt = find_latest_checkpoint(sft_ckpt_dir, "stage3_vanilla_sft")

    if latest_ckpt and "step_400000" in latest_ckpt.name:
        print("[Orchestrator] Vanilla SFT has reached step 400,000! Skipping to next stage.")
        return

    print("\n" + "=" * 80)
    if latest_ckpt:
        print(f"[Orchestrator] Resuming Vanilla SFT from: {latest_ckpt}")
    else:
        print("[Orchestrator] Launching Vanilla SFT on ENTIRE Datasets pool (including Common Voice)...")
    print("=" * 80 + "\n")

    header = "🚀 <b>ABLATION STUDY: RESTARTING VANILLA SFT FROM SCRATCH (FULL 5-SOURCE POOL - 400K TARGET)</b>" if not latest_ckpt else "🚀 <b>ABLATION STUDY: RESUMING VANILLA SFT (400K TARGET)</b>"
    send_telegram(f"""{header}
• <b>Model:</b> <code>scaled-conformer512-moe-3stage</code> (130.8M MoE)
• <b>Starting State:</b> <code>{'Fresh Start (Step 0 / 400,000)' if not latest_ckpt else latest_ckpt.name}</code>
• <b>Datasets Pool (5 sources):</b>
  - 100% RESPIN (809,934 utts)
  - OpenSLR64 Marathi (1,569 utts)
  - Kathbath Clean + Noisy (85,701 utts)
  - <b>Mozilla Common Voice Marathi (1,205 utts)</b>
  - Pretraining Relayed Stream
• <b>Target:</b> 400,000 steps (Batch 32, Grad Accum 4 = 128 effective)
• <b>LR:</b> 3e-5 → 5e-6 (Warmup 1,000 steps, Cosine Decay)""")

    stage2_ckpt = RUN_DIR / "checkpoints" / "stage2_moe" / "best_model.pt"
    cmd = [
        str(PYTHON_EXE), "-u", "final_alignment/train_alignment.py",
        "--run_dir", str(RUN_DIR),
        "--moe_ckpt", str(stage2_ckpt),
        "--checkpoint_dir", str(sft_ckpt_dir),
        "--stage_name", "stage3_vanilla_sft",
        "--steps", "400000",
        "--batch_size", "32",
        "--grad_accum_steps", "4",
        "--lr", "3e-5",
        "--min_lr", "5e-6",
        "--warmup_steps", "1000",
        "--log_every", "25",
        "--save_every", "2000",
    ]
    if latest_ckpt:
        cmd.extend(["--resume", str(latest_ckpt)])

    subprocess.run(cmd, cwd=str(Path(r"D:\marathi-asr")), check=True)


def run_grpo_polish():
    grpo_ckpt_dir = RUN_DIR / "checkpoints" / "stage3_grpo_polish"
    best_grpo = grpo_ckpt_dir / "best_model.pt"
    if best_grpo.exists():
        print(f"[Orchestrator] GRPO Polish checkpoint already exists at {best_grpo}. Skipping training.")
        return

    sft_ckpt = RUN_DIR / "checkpoints" / "stage3_vanilla_sft" / "best_model.pt"
    if not sft_ckpt.exists():
        raise FileNotFoundError(f"Could not find SFT checkpoint at {sft_ckpt}")

    print("\n" + "=" * 80)
    print("[Orchestrator] Launching GRPO Polish on ENTIRE Datasets...")
    print("=" * 80 + "\n")

    send_telegram("""⚡ <b>ABLATION STUDY: GRPO SEQUENCE POLISH LAUNCHED</b>
• <b>Reference / Starting Model:</b> Vanilla SFT <code>best_model.pt</code>
• <b>Objective:</b> Direct Sequence CER/WER Minimization via GRPO
• <b>Group Size:</b> G=4 rollouts per utterance (1 greedy + 3 stochastic)
• <b>Datasets:</b> 100% RESPIN + OpenSLR64 + Kathbath Clean/Noisy + Pretrain Stream
• <b>Steps:</b> 5,000 steps (Batch 8, Grad Accum 2 = 16 effective utts * 4 rollouts)
• <b>LR:</b> 2e-5 → 2e-6""")

    cmd = [
        str(PYTHON_EXE), "-u", "final_alignment/train_grpo_polish.py",
        "--run_dir", str(RUN_DIR),
        "--moe_ckpt", str(sft_ckpt),
        "--checkpoint_dir", str(grpo_ckpt_dir),
        "--stage_name", "stage3_grpo_polish",
        "--steps", "5000",
        "--batch_size", "8",
        "--group_size", "4",
        "--temperature", "0.85",
        "--gamma_wer", "0.20",
        "--clip_eps", "0.20",
        "--beta_kl", "0.04",
        "--alpha_ctc", "0.15",
        "--lr", "2e-5",
        "--min_lr", "2e-6",
        "--warmup_steps", "150",
        "--grad_accum_steps", "2",
        "--log_every", "10",
        "--plot_every", "25",
        "--save_every", "500",
    ]
    subprocess.run(cmd, cwd=str(Path(r"D:\marathi-asr")), check=True)


def run_comparative_benchmark():
    sft_ckpt = RUN_DIR / "checkpoints" / "stage3_vanilla_sft" / "best_model.pt"
    grpo_ckpt = RUN_DIR / "checkpoints" / "stage3_grpo_polish" / "best_model.pt"

    print("\n" + "=" * 80)
    print("[Orchestrator] Running GRPO vs. Vanilla SFT Benchmark on Frozen RESPIN Test Set...")
    print("=" * 80 + "\n")

    cmd = [
        str(PYTHON_EXE), "inference/grpo_vs_sft/benchmark_grpo_vs_sft.py",
        "--sft_checkpoint", str(sft_ckpt),
        "--grpo_checkpoint", str(grpo_ckpt),
        "--output_dir", "inference/grpo_vs_sft/results",
        "--beam_width", "16",
    ]
    subprocess.run(cmd, cwd=str(Path(r"D:\marathi-asr")), check=True)

    send_telegram("""🎉 <b>GRPO VS. VANILLA SFT ABLATION STUDY COMPLETE!</b>
• Frozen RESPIN test set evaluated across D1, D2, D3, D4.
• Paired statistical significance tests computed.
• Summary, Markdown reports, and LaTeX tables generated in:
  <code>inference/grpo_vs_sft/results/</code>""")


def main():
    print("[Orchestrator] Starting Sequential GRPO vs. Vanilla SFT Ablation Pipeline...")
    run_vanilla_sft()
    run_grpo_polish()
    run_comparative_benchmark()
    print("[Orchestrator] All ablation tasks finished successfully!")


if __name__ == "__main__":
    main()
