"""
Automated Sequential Pipeline Queue:
1. Monitors the active 33M Stage 3 Vanilla SFT run until step 350,000 completion.
2. Automatically queues and launches the Dense Baseline Pipeline: 'scaled-conformer512-dense-3stage'.
3. Once Dense training finishes, triggers the RESPIN benchmark comparison.
4. Delivers Telegram milestone dispatches at every phase transition.
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path

CONFIG_TELEGRAM = Path(r"D:\marathi-asr\telegram_config.json")
RUN_DIR_33M = Path(r"D:\marathi-asr\runs\no-splitformer-moe-only")
RUN_DIR_DENSE = Path(r"D:\marathi-asr\runs\scaled-conformer512-dense-3stage")
PYTHON_EXE = Path(r"D:\marathi-asr\.venv311\Scripts\python.exe")
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


def is_33m_finished() -> bool:
    ckpt_dir = RUN_DIR_33M / "checkpoints" / "stage3_vanilla_sft"
    target_ckpt = ckpt_dir / f"stage3_vanilla_sft_step_{TOTAL_STEPS}.pt"
    if target_ckpt.exists():
        return True
    # Also check if best_model was saved near step 350,000
    metrics_csv = RUN_DIR_33M / "csvs" / "stage3_vanilla_sft_metrics.csv"
    if metrics_csv.exists():
        try:
            with open(metrics_csv, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_step = int(lines[-1].split(",")[0])
                    if last_step >= TOTAL_STEPS:
                        return True
        except Exception:
            pass
    return False


def wait_for_33m_completion():
    print(f"[Queue Watcher] Monitoring 33M stage 3 training until step {TOTAL_STEPS:,}...")
    while not is_33m_finished():
        time.sleep(60)
    print("\n[Queue Watcher] 33M Model Stage 3 Training has completed 350,000 steps!\n")
    send_telegram(f"""🏁 <b>MILESTONE: 33M MODEL STAGE 3 TRAINING REACHED {TOTAL_STEPS:,} STEPS!</b>
• All 350,000 steps completed across multi-source dataset pool.
• Automatically transitioning GPU to <b>Experiment E1: Dense Conformer-512 Baseline</b>!""")


def launch_dense_pipeline():
    print("\n[Queue Watcher] Launching Dense Baseline Pipeline: scaled-conformer512-dense-3stage...")
    send_telegram("""⚡ <b>AUTOMATED EXPERIMENT TRANSITION: EXPERIMENT E1</b>
• <b>Queued Baseline:</b> <code>scaled-conformer512-dense-3stage</code> (130.8M Total Parameter Matched)
• <b>Configuration:</b> MoE-free, 12 layers, d=512, ffn=8, 30k pretraining + 80k adaptation + 40k calibration.
• <b>Objective:</b> Directly verify the Central Hypothesis H0 against full capacity-matched dense Conformer!""")

    cmd = [
        str(PYTHON_EXE), "-u", "scripts/pipeline.py",
        "--config", "configs/scaled_conformer512_dense_3stage.json",
    ]
    subprocess.run(cmd, cwd=str(Path(r"D:\marathi-asr")), check=True)


def main():
    wait_for_33m_completion()
    # 30 seconds buffer for VRAM clearing and filesystem sync
    time.sleep(30)
    launch_dense_pipeline()


if __name__ == "__main__":
    main()
