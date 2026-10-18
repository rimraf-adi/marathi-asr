"""
Automated Sequential Experiment Orchestrator:
1. Monitors the active MoE training run ('scaled-conformer512-moe-3stage') until completion.
2. Automatically launches the Dense Baseline run ('scaled-conformer512-dense-3stage').
3. Once Dense training finishes, triggers the 'inference/moe_vs_notmoe/benchmark_moe_vs_dense.py' benchmark on the RESPIN test set.
4. Delivers Telegram milestone updates at every key transition.
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
RUN_DIR_MOE = Path(r"D:\marathi-asr\runs\scaled-conformer512-moe-3stage")
RUN_DIR_DENSE = Path(r"D:\marathi-asr\runs\scaled-conformer512-dense-3stage")
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


def is_moe_run_finished() -> bool:
    status_file = RUN_DIR_MOE / "pipeline_status.json"
    if not status_file.exists():
        return False
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        s3_done = data.get("stage3_final_alignment", {}).get("status") == "completed"
        s4_done = data.get("final_benchmark", {}).get("status") == "completed"
        return s3_done and s4_done
    except Exception:
        return False


def wait_for_moe_completion():
    print("[Orchestrator] Monitoring 'scaled-conformer512-moe-3stage' completion...")
    while not is_moe_run_finished():
        time.sleep(60)
    print("\n[Orchestrator] MoE Run has completed Stage 3 Final Alignment!\n")


def launch_dense_pipeline():
    print("\n[Orchestrator] Launching Dense Baseline Pipeline: scaled-conformer512-dense-3stage...")
    send_telegram("""⚡ <b>AUTOMATED PIPELINE TRANSITION</b>
• <b>MoE Run:</b> Stage 3 Completed!
• <b>Next Action:</b> Automatically launching <b>Dense Conformer Baseline</b> (<code>scaled-conformer512-dense-3stage</code>).
• <b>Configuration:</b> MoE-free, 30k pretraining steps, pooled RESPIN adaptation (80k steps), multi-source calibration (40k steps).
• <b>Target:</b> Directly test the Negative Transfer hypothesis against MoE on frozen RESPIN test set!""")

    cmd = [
        str(PYTHON_EXE), "-u", "scripts/pipeline.py",
        "--config", "configs/scaled_conformer512_dense_3stage.json",
    ]
    subprocess.run(cmd, cwd=str(Path(r"D:\marathi-asr")), check=True)


def run_final_comparative_benchmark():
    print("\n[Orchestrator] Running Final MoE vs. Dense Benchmark on RESPIN Test Set...")
    cmd = [
        str(PYTHON_EXE), "inference/moe_vs_notmoe/benchmark_moe_vs_dense.py",
        "--moe_checkpoint", str(RUN_DIR_MOE / "checkpoints/stage3_final_alignment/best_model.pt"),
        "--dense_checkpoint", str(RUN_DIR_DENSE / "checkpoints/stage3_final_alignment/best_model.pt"),
        "--output_dir", "inference/moe_vs_notmoe/results",
    ]
    subprocess.run(cmd, cwd=str(Path(r"D:\marathi-asr")), check=True)

    send_telegram("""🎉 <b>MOE VS. DENSE COMPARATIVE BENCHMARK COMPLETE!</b>
• Frozen RESPIN test set evaluated across D1, D2, D3, D4.
• Statistical hypothesis testing completed.
• Reports generated in <code>inference/moe_vs_notmoe/results/</code>!""")


def main():
    wait_for_moe_completion()
    # Allow 15 seconds for filesystem settle
    time.sleep(15)
    launch_dense_pipeline()
    run_final_comparative_benchmark()


if __name__ == "__main__":
    main()
