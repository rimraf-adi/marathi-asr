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
from datetime import datetime, timedelta

CONFIG_CANDIDATES = [
    Path(__file__).parent / "telegram_config.json",
    Path(__file__).parent.parent / "telegram_config.json",
    Path(r"D:\marathi-asr\telegram_config.json"),
    Path(r"D:\marathi-asr\scripts\telegram_config.json"),
]

CANDIDATE_RUN_DIRS = [
    Path(r"D:\marathi-asr\runs\no-splitformer-moe-only"),
    Path(r"D:\marathi-asr\runs\scaled-conformer512-dense-3stage"),
    Path(r"D:\marathi-asr\runs\scaled-conformer512-moe-3stage"),
]

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
    for upd in reversed(updates):
        msg = upd.get("message") or upd.get("channel_post") or upd.get("my_chat_member")
        if msg:
            chat = msg.get("chat")
            if chat and "id" in chat:
                return chat["id"]
    return None

def send_message(token, chat_id, text, parse_mode="HTML"):
    data = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true"
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    res = telegram_api(token, "sendMessage", data)
    # If parse error, fallback to plain text
    if not res or not res.get("ok"):
        if parse_mode:
            data.pop("parse_mode", None)
            res = telegram_api(token, "sendMessage", data)
    return res

def get_active_run_dir():
    best_dir = CANDIDATE_RUN_DIRS[0]
    best_time = 0.0
    for rdir in CANDIDATE_RUN_DIRS:
        if not rdir.exists():
            continue
        # Check logs and checkpoints
        for f in rdir.glob("logs/*telemetry*.jsonl"):
            try:
                mtime = f.stat().st_mtime
                if mtime > best_time:
                    best_time = mtime
                    best_dir = rdir
            except Exception:
                pass
        for f in rdir.glob("checkpoints/*/*.pt"):
            try:
                mtime = f.stat().st_mtime
                if mtime > best_time:
                    best_time = mtime
                    best_dir = rdir
            except Exception:
                pass
    return best_dir

def get_latest_telemetry_and_history(run_dir: Path):
    telemetry_files = sorted(run_dir.glob("logs/*telemetry*.jsonl"), key=os.path.getmtime, reverse=True)
    if not telemetry_files:
        return None, []
    latest_file = telemetry_files[0]
    lines = []
    with open(latest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    lines.append(json.loads(line.strip()))
                except Exception:
                    pass
    if not lines:
        return None, []
    return lines[-1], lines

def get_latest_checkpoint(run_dir: Path):
    for sub in ["stage3_vanilla_sft", "stage3_final_alignment", "stage2_dense", "stage2_moe", "stage1_pretrain", "checkpoints"]:
        ckpt_dir = run_dir / "checkpoints" / sub
        if not ckpt_dir.exists():
            ckpt_dir = run_dir / sub
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
            if ckpts:
                size_mb = ckpts[0].stat().st_size / (1024 * 1024)
                return f"{sub}/{ckpts[0].name} ({size_mb:.1f} MB)"
    return "None"

def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "vram_used_gb": float(parts[0]) / 1024.0,
            "vram_total_gb": float(parts[1]) / 1024.0,
            "util": parts[2],
            "temp": parts[3],
            "power": parts[4] if len(parts) > 4 else "N/A"
        }
    except Exception:
        return None

def get_disk_free():
    try:
        usage = shutil.disk_usage("D:\\")
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0

def is_process_running(keyword: str):
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", f"Get-Process | Where-Object {{ $_.CommandLine -like '*{keyword}*' }} | Measure-Object | Select-Object -ExpandProperty Count"],
            text=True
        ).strip()
        return int(out) > 0
    except Exception:
        # Fallback check
        return False

def build_status_message(telemetry, history, ckpt_name, gpu, disk_free, run_dir):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = run_dir.name
    
    # Check if pretraining or supervised
    is_pretrain = "masked_l1" in (telemetry or {})
    
    # Determine total steps & stage phase
    if is_pretrain:
        total_steps = 30000
        phase = "Stage 1: Pretraining (Dense Conformer-512 Backbone)"
    elif "stage2_dense" in ckpt_name or (run_dir / "logs" / "stage2_dense.log").exists():
        total_steps = 80000
        phase = "Stage 2: Supervised CTC Adaptation (Dense Baseline)"
    elif "stage3_final_alignment" in ckpt_name:
        total_steps = 40000
        phase = "Stage 3: Multi-Source Calibration (Dense Baseline)"
    elif "stage3_vanilla_sft" in ckpt_name or "no-splitformer" in run_name:
        total_steps = 350000
        phase = "Stage 3 Vanilla SFT (Calibration)"
    else:
        total_steps = 60000
        phase = "Active Training Phase"

    step = telemetry.get("step", 0) if telemetry else 0
    if step > total_steps and not is_pretrain:
        total_steps = step
    pct = (step / total_steps) * 100 if total_steps else 0.0

    throughput = telemetry.get("throughput_aud_s_per_s", 0.0) if telemetry else 0.0
    lr = telemetry.get("lr", 0.0) if telemetry else 0.0

    # Calculate step speed & ETA
    eta_str = "Calculating..."
    steps_per_min = 0.0
    if len(history) >= 10:
        recent = history[-10:]
        dt = recent[-1].get("timestamp", 0) - recent[0].get("timestamp", 0)
        dstep = recent[-1].get("step", 0) - recent[0].get("step", 0)
        if dt > 0 and dstep > 0:
            rate = dstep / dt  # steps per sec
            steps_per_min = rate * 60.0
            remaining_steps = max(0, total_steps - step)
            rem_sec = remaining_steps / rate if rate > 0 else 0
            if rem_sec > 0:
                hours = int(rem_sec // 3600)
                minutes = int((rem_sec % 3600) // 60)
                eta_str = f"~{hours}h {minutes}m ({remaining_steps:,} steps left)"
            else:
                eta_str = "Stage target reached!"

    gpu_str = "N/A"
    if gpu:
        gpu_str = f"RTX A5000 | <code>{gpu['vram_used_gb']:.1f} / {gpu['vram_total_gb']:.1f} GB</code> | <code>{gpu['temp']}°C</code> | <code>{gpu['util']}% util</code> ({gpu['power']}W)"

    # Format metric block depending on pretrain vs supervised
    if is_pretrain:
        masked_l1 = telemetry.get("masked_l1", 0.0) if telemetry else 0.0
        snr_db = telemetry.get("snr_db", 0.0) if telemetry else 0.0
        eff_rank = telemetry.get("effective_rank", 0.0) if telemetry else 0.0
        epoch = telemetry.get("epoch", 1) if telemetry else 1
        metrics_block = (
            f"🔄 <b>Epoch:</b> <code>{epoch} / 10</code>\n"
            f"📉 <b>Masked L1 Loss:</b> <code>{masked_l1:.4f}</code>\n"
            f"🔊 <b>Spectral SNR:</b> <code>{snr_db:.2f} dB</code>\n"
            f"🧬 <b>Effective Rank:</b> <code>{eff_rank:.1f}</code>\n"
        )
    else:
        ctc_loss = telemetry.get("ctc_loss", 0.0) if telemetry else 0.0
        aux_loss = telemetry.get("aux_loss")
        cer = telemetry.get("cer", 0.0) if telemetry else 0.0
        router_line = f"• <b>Router Aux Loss:</b> <code>{aux_loss:.4f}</code>\n" if aux_loss is not None else ""
        metrics_block = (
            f"📉 <b>CTC Loss:</b> <code>{ctc_loss:.4f}</code>\n" +
            router_line +
            f"🎯 <b>Batch CER:</b> <code>{cer*100:.2f}%</code>\n"
        )

    # Status of experiment
    exp_status = "🎯 <b>Active Experiment:</b> <code>Experiment E1 (Dense Conformer-512 Baseline, 130.8M Total Param Matched)</code>" if "dense" in run_name else "🎯 <b>Active Experiment:</b> <code>33M Sparse MoE Production Calibrated Model</code>"

    msg = (
        f"🤖 <b>Marathi ASR Training Telemetry Update</b>\n"
        f"🕒 <b>Timestamp:</b> <code>{now_str}</code>\n"
        f"📁 <b>Active Run:</b> <code>{run_name}</code>\n\n"
        f"{exp_status}\n"
        f"🎯 <b>Phase:</b> <code>{phase}</code>\n"
        f"📊 <b>Progress:</b> <b>{step:,} / {total_steps:,} steps</b> (<b>{pct:.1f}%</b>)\n"
        f"⏳ <b>ETA:</b> <code>{eta_str}</code>" + (f" (<code>{steps_per_min:.1f} steps/min</code>)\n" if steps_per_min > 0 else "\n") +
        metrics_block +
        f"⚡ <b>Throughput:</b> <code>{throughput:.1f} aud-s/s</code>\n"
        f"📈 <b>Learning Rate:</b> <code>{lr:.2e}</code>\n"
        f"💾 <b>Checkpoint:</b> <code>{ckpt_name}</code>\n\n"
        f"🖥️ <b>GPU:</b> {gpu_str}\n"
        f"💿 <b>Disk Free:</b> <code>{disk_free:.1f} GB</code> on D:"
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
            send_message(token, chat_id, "✅ <b>Telegram notifications linked successfully!</b>\nYou will receive training updates here.")
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
        run_dir = get_active_run_dir()
        telemetry, history = get_latest_telemetry_and_history(run_dir)
        ckpt_name = get_latest_checkpoint(run_dir)
        gpu = get_gpu_info()
        disk_free = get_disk_free()
        msg = build_status_message(telemetry, history, ckpt_name, gpu, disk_free, run_dir)

    # Persist latest text message
    txt_path = Path(__file__).parent / "latest_telegram_msg.txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass

    res = send_message(token, chat_id, msg, parse_mode="HTML")
    if res and res.get("ok"):
        print("Telegram update sent successfully.")
    else:
        print(f"Failed to send message: {res}")

if __name__ == "__main__":
    main()
