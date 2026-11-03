"""Background monitor for the Marathi ASR ablation queue.
Sends Telegram telemetry updates every 2 hours (and on milestone completions or failures).
"""
import json
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\marathi-asr")
RUN_DIR = ROOT / "runs" / "ablations"
EVALS_DIR = ROOT / "evals" / "ablations"
TELEGRAM_SCRIPT = ROOT / "send_telegram_update.py"

TOTAL_S1_STEPS = 18141
TOTAL_VAR_STEPS = 20000
TOTAL_VARIANTS = 9
VARIANTS = [
    "base_s0", "base_s1", "base_s2", "scratch_s0",
    "k1_s0", "k2_s0", "k4_s0", "lower4_s0", "all12_s0"
]

def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "vram_used": float(parts[0]) / 1024.0,
            "vram_total": float(parts[1]) / 1024.0,
            "util": parts[2],
            "temp": parts[3],
            "power": parts[4] if len(parts) > 4 else "N/A"
        }
    except Exception:
        return None

def get_disk_free():
    try:
        return shutil.disk_usage("D:\\").free / (1024 ** 3)
    except Exception:
        return 0.0

def get_finished_variants():
    if not EVALS_DIR.exists():
        return []
    return [p.stem for p in EVALS_DIR.glob("*.json")]

def get_active_stage():
    s1_done = RUN_DIR / "stage1" / "DONE"
    if not s1_done.exists():
        # Stage 1 active
        log_file = RUN_DIR / "stage1" / "logs" / "stage1_pretrain_telemetry.jsonl"
        last_item = None
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            last_item = json.loads(line)
            except Exception:
                pass
        return "stage1", last_item
    
    # Check variants
    for v in VARIANTS:
        v_done = RUN_DIR / v / "train.done"
        if not v_done.exists():
            # Variant v is training or about to train
            log_files = sorted(RUN_DIR.glob(f"{v}/logs/*telemetry*.jsonl"), key=os.path.getmtime, reverse=True)
            last_item = None
            if log_files and log_files[0].exists():
                try:
                    with open(log_files[0], "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                last_item = json.loads(line)
                except Exception:
                    pass
            return v, last_item
            
    return "done", None

def format_update_message():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu = get_gpu_info()
    disk_free = get_disk_free()
    finished = get_finished_variants()
    active_name, telemetry = get_active_stage()
    
    gpu_str = f"RTX A5000 | <code>{gpu['vram_used']:.1f} / {gpu['vram_total']:.1f} GB</code> | <code>{gpu['temp']}°C</code> | <code>{gpu['util']}% util</code>" if gpu else "N/A"
    finished_str = ", ".join(finished) if finished else "None yet (Stage 1 in progress)"
    
    if active_name == "stage1":
        step = telemetry.get("step", 0) if telemetry else 0
        loss = telemetry.get("masked_l1", 0.0) if telemetry else 0.0
        pct = (step / TOTAL_S1_STEPS) * 100 if TOTAL_S1_STEPS else 0.0
        lr = telemetry.get("lr", 0.0) if telemetry else 0.0
        
        body = (
            f"🔄 <b>Phase:</b> <code>Phase 0: Stage 1 Pretraining</code>\n"
            f"📊 <b>Progress:</b> <b>{step:,} / {TOTAL_S1_STEPS:,} steps</b> (<b>{pct:.1f}%</b>)\n"
            f"📉 <b>Masked L1 Loss:</b> <code>{loss:.4f}</code>\n"
            f"📈 <b>Learning Rate:</b> <code>{lr:.2e}</code>\n"
        )
    elif active_name == "done":
        body = (
            f"🎉 <b>Status: ALL ABLATION RUNS COMPLETED!</b>\n"
            f"All {len(finished)} / {TOTAL_VARIANTS} variants evaluated.\n"
        )
    else:
        step = telemetry.get("step", 0) if telemetry else 0
        pct = (step / TOTAL_VAR_STEPS) * 100 if TOTAL_VAR_STEPS else 0.0
        loss = telemetry.get("ctc_loss", 0.0) if telemetry else 0.0
        aux = telemetry.get("aux_loss")
        aux_str = f"• <b>Aux Loss:</b> <code>{aux:.4f}</code>\n" if aux is not None else ""
        
        body = (
            f"🔬 <b>Variant:</b> <code>{active_name}</code>\n"
            f"📊 <b>Progress:</b> <b>{step:,} / {TOTAL_VAR_STEPS:,} steps</b> (<b>{pct:.1f}%</b>)\n"
            f"📉 <b>CTC Loss:</b> <code>{loss:.4f}</code>\n"
            f"{aux_str}"
        )

    msg = (
        f"🤖 <b>Marathi ASR Ablation Queue Update</b>\n"
        f"🕒 <code>{now_str}</code>\n\n"
        f"{body}\n"
        f"📁 <b>Completed Variants ({len(finished)}/{TOTAL_VARIANTS}):</b>\n<code>{finished_str}</code>\n\n"
        f"🖥️ <b>GPU:</b> {gpu_str}\n"
        f"💿 <b>Disk Free:</b> <code>{disk_free:.1f} GB</code>"
    )
    return msg

def send_telegram(msg):
    try:
        py_exe = ROOT / ".venv311" / "Scripts" / "python.exe"
        subprocess.run(
            [str(py_exe), str(TELEGRAM_SCRIPT), "-m", msg],
            cwd=str(ROOT),
            check=True,
            timeout=30
        )
        print(f"[{datetime.now()}] Sent Telegram update.")
    except Exception as e:
        print(f"[{datetime.now()}] Error sending Telegram update: {e}", file=sys.stderr)

def main():
    print(f"Starting Ablation Monitor. Interval: 2 hours.")
    INTERVAL = 7200  # 2 hours in seconds
    
    while True:
        try:
            time.sleep(INTERVAL)
            msg = format_update_message()
            send_telegram(msg)
            
            # Check if all done
            active_name, _ = get_active_stage()
            if active_name == "done":
                print("All ablations complete. Monitor exiting.")
                break
        except Exception as e:
            print(f"Monitor loop error: {e}", file=sys.stderr)
            time.sleep(60)

if __name__ == "__main__":
    main()
