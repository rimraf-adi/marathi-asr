"""
Manual Telegram Telemetry Notifier.
Composes rich, human-readable training updates without rigid robotic templates.
Reads active run state, GPU metrics, loss dynamics, and delivers updates via Telegram bot.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
import pandas as pd
import subprocess

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")
RUN_DIR = Path(r"D:\marathi-asr\runs\scaled-conformer512-moe-3stage")


def get_gpu_info():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, text=True).strip().split(",")
        return {
            "util": float(out[0].strip()),
            "mem_used": float(out[1].strip()) / 1024.0,
            "mem_total": float(out[2].strip()) / 1024.0,
            "temp": int(out[3].strip()),
            "power": float(out[4].strip())
        }
    except Exception:
        return None


def get_latest_stage_info():
    status_file = RUN_DIR / "pipeline_status.json"
    active_stage = "stage1_pretrain"
    elapsed_str = "N/A"
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for st, info in data.items():
                    if info.get("status") == "running":
                        active_stage = st
                        updated = info.get("timestamp", time.time())
                        elapsed_min = (time.time() - updated) / 60.0
                        elapsed_str = f"{elapsed_min:.1f} mins"
        except Exception:
            pass

    # Find latest CSV metrics
    csvs = list(RUN_DIR.glob("csvs/*.csv"))
    latest_metrics = {}
    if csvs:
        # Sort by modification time
        csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_csv = csvs[0]
        try:
            df = pd.read_csv(latest_csv)
            if not df.empty:
                latest_metrics = df.iloc[-1].to_dict()
        except Exception:
            pass

    return active_stage, elapsed_str, latest_metrics


def send_telegram(text: str) -> bool:
    if not CONFIG_PATH.exists():
        print("Telegram config not found!")
        return False
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    token = cfg.get("bot_token")
    chat_id = cfg.get("chat_id")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res.get("ok", False)
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")
        return False


def build_and_send_update(custom_note: str = ""):
    stage, elapsed, metrics = get_latest_stage_info()
    gpu = get_gpu_info()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"📊 <b>Marathi ASR Training Update ({now_str})</b>",
        f"<b>Active Stage:</b> <code>{stage}</code> (Running for ~{elapsed})",
        f"<b>Run:</b> <code>{RUN_DIR.name}</code> (Conformer-512 MoE)",
        "",
    ]

    # Metrics section
    if metrics:
        step = metrics.get("step", "N/A")
        lines.append(f"<b>Progress:</b> Step {step:,}" if isinstance(step, (int, float)) else f"<b>Progress:</b> Step {step}")
        
        if "loss_masked_l1" in metrics:
            lines.append(f"• <b>Masked Reconstruction L1 Loss:</b> {metrics['loss_masked_l1']:.4f}")
        if "reconstruction_snr_db" in metrics:
            lines.append(f"• <b>Reconstruction SNR:</b> {metrics['reconstruction_snr_db']:.2f} dB")
        if "ctc_loss" in metrics:
            lines.append(f"• <b>CTC Loss:</b> {metrics['ctc_loss']:.4f}")
        if "cer" in metrics:
            lines.append(f"• <b>Current Greedy CER:</b> {metrics['cer'] * 100:.2f}%")
        if "throughput_aud_s_per_s" in metrics:
            lines.append(f"• <b>Throughput:</b> {metrics['throughput_aud_s_per_s']:.1f} audio-sec/sec")
        if "total_audio_hours" in metrics:
            lines.append(f"• <b>Audio Processed:</b> {metrics['total_audio_hours']:.1f} hours")
        lines.append("")
    else:
        lines.append("<i>Dataset streaming and local chunk ingestion actively in progress...</i>\n")

    # Hardware health
    if gpu:
        lines.append("<b>Hardware Vitals (RTX A5000):</b>")
        lines.append(f"• <b>VRAM Usage:</b> {gpu['mem_used']:.2f} GB / {gpu['mem_total']:.2f} GB ({(gpu['mem_used']/gpu['mem_total'])*100:.1f}%)")
        lines.append(f"• <b>GPU Temp:</b> {gpu['temp']}°C | <b>Power:</b> {gpu['power']:.1f} W | <b>Load:</b> {gpu['util']:.0f}%")
        lines.append("")

    if custom_note:
        lines.append(f"<b>Note:</b> {custom_note}\n")

    lines.append("<i>Next automatic report in 2 hours.</i>")
    message = "\n".join(lines)
    return send_telegram(message)


if __name__ == "__main__":
    note = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    success = build_and_send_update(note)
    print("Notification sent:", success)
