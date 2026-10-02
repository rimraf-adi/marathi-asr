import json
import os
import sys
import argparse
import glob
import time
import shutil
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime

CONFIG_CANDIDATES = [
    Path(__file__).parent / "telegram_config.json",
    Path(__file__).parent.parent / "telegram_config.json",
    Path(r"D:\marathi-asr\telegram_config.json"),
    Path(r"D:\marathi-asr\scripts\telegram_config.json"),
]
RUN_DIR = Path(r"D:\marathi-asr\runs\no-splitformer-moe-only")

def load_config():
    for p in CONFIG_CANDIDATES:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

def save_config(cfg):
    target = CONFIG_CANDIDATES[0]
    with open(target, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def telegram_api(token, method, data=None):
    url = f"https://api.telegram.org/bot{token}/{method}"
    if data:
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=encoded, headers={"Content-Type": "application/x-www-form-urlencoded"})
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[Telegram API Error] {e}")
        return None

def find_chat_id(token):
    res = telegram_api(token, "getUpdates")
    if not res or not res.get("ok"):
        return None
    updates = res.get("result", [])
    if not updates:
        return None
    # Pick latest message
    for upd in reversed(updates):
        msg = upd.get("message") or upd.get("channel_post") or upd.get("my_chat_member")
        if msg:
            chat = msg.get("chat")
            if chat and "id" in chat:
                return chat["id"]
    return None

def send_message(token, chat_id, text, parse_mode="Markdown"):
    data = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true"
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    res = telegram_api(token, "sendMessage", data)
    # If Markdown parsing failed (e.g. unescaped underscores/asterisks), retry as plain text
    if not res or not res.get("ok"):
        if parse_mode:
            data.pop("parse_mode", None)
            res = telegram_api(token, "sendMessage", data)
    return res

def get_latest_telemetry():
    telemetry_files = sorted(RUN_DIR.glob("logs/*telemetry*.jsonl"), key=os.path.getmtime, reverse=True)
    if not telemetry_files:
        return None
    latest_file = telemetry_files[0]
    last_line = None
    with open(latest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    if last_line:
        try:
            return json.loads(last_line)
        except Exception:
            return None
    return None

def get_latest_checkpoint():
    ckpt_dir = RUN_DIR / "checkpoints" / "stage2_moe"
    if not ckpt_dir.exists():
        return "None"
    ckpts = sorted(ckpt_dir.glob("stage2_moe_step_*.pt"), key=os.path.getmtime, reverse=True)
    if ckpts:
        return ckpts[0].name
    return "None"

def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "vram_used_gb": float(parts[0]) / 1024.0,
            "vram_total_gb": float(parts[1]) / 1024.0,
            "util": parts[2],
            "temp": parts[3]
        }
    except Exception:
        return None

def get_disk_free():
    try:
        usage = shutil.disk_usage("D:\\")
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0

def build_status_message(telemetry, ckpt_name, gpu, disk_free):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_steps = 60000
    step = telemetry.get("step", 0) if telemetry else 0
    pct = (step / total_steps) * 100 if total_steps else 0.0
    phase = telemetry.get("phase", "Unknown") if telemetry else "Unknown"
    ctc_loss = telemetry.get("ctc_loss", 0.0) if telemetry else 0.0
    aux_loss = telemetry.get("aux_loss", 0.0) if telemetry else 0.0
    cer = telemetry.get("cer", 0.0) if telemetry else 0.0
    throughput = telemetry.get("throughput_aud_s_per_s", 0.0) if telemetry else 0.0
    lr = telemetry.get("lr", 0.0) if telemetry else 0.0

    gpu_str = "N/A"
    if gpu:
        gpu_str = f"RTX A5000 | `{gpu['vram_used_gb']:.1f} / {gpu['vram_total_gb']:.1f} GB` | `{gpu['temp']}°C` ({gpu['util']}% util)"

    msg = (
        f"🤖 *Marathi ASR Training Update*\n"
        f"🕒 *Time*: `{now_str}`\n\n"
        f"📊 *Progress*: Step *{step:,} / {total_steps:,}* ({pct:.1f}%)\n"
        f"🎯 *Phase*: `{phase}`\n"
        f"📉 *CTC Loss*: `{ctc_loss:.4f}`\n"
        f"⚖️ *Router Loss*: `{aux_loss:.4f}`\n"
        f"🎯 *Batch CER*: `{cer*100:.2f}%`\n"
        f"⚡ *Throughput*: `{throughput:.1f} aud-s/s`\n"
        f"📈 *LR*: `{lr:.2e}`\n"
        f"💾 *Latest Checkpoint*: `{ckpt_name}`\n\n"
        f"🖥️ *GPU*: {gpu_str}\n"
        f"💿 *Disk Free*: `{disk_free:.1f} GB free` on D:"
    )
    return msg

def main():
    parser = argparse.ArgumentParser(description="Send Telegram Training Update")
    parser.add_argument("-m", "--message", type=str, default=None, help="Manual status message to send")
    parser.add_argument("-f", "--file", type=str, default=None, help="Path to text file containing message")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg or not cfg.get("bot_token"):
        print("Config missing or no bot_token specified.")
        sys.exit(1)
        
    token = cfg["bot_token"]
    chat_id = cfg.get("chat_id")
    
    # Check if chat_id needs discovery
    if not chat_id:
        print("Searching for chat_id from Telegram getUpdates...")
        chat_id = find_chat_id(token)
        if chat_id:
            cfg["chat_id"] = chat_id
            save_config(cfg)
            print(f"Discovered chat_id: {chat_id}. Saved to config.")
            send_message(token, chat_id, "✅ *Telegram notifications linked successfully!*\nYou will receive training updates here.")
        else:
            print("No chat_id found yet. Please open Telegram and send /start or any message to your bot: @fypupdates_bot")
            sys.exit(0)

    # Determine message content
    if args.message:
        msg = args.message
    elif args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            msg = f.read().strip()
    else:
        # Fallback to gathered metrics
        telemetry = get_latest_telemetry()
        ckpt_name = get_latest_checkpoint()
        gpu = get_gpu_info()
        disk_free = get_disk_free()
        msg = build_status_message(telemetry, ckpt_name, gpu, disk_free)

    res = send_message(token, chat_id, msg)
    if res and res.get("ok"):
        print("Telegram update sent successfully.")
    else:
        print(f"Failed to send message: {res}")

if __name__ == "__main__":
    main()
