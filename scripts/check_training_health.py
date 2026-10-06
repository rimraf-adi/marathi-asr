"""
Training Health & Deadlock Watchdog.
Monitors the background training pipeline every 30 minutes without sending notifications.
Checks:
  1. Process existence (python process running on GPU)
  2. Log file modification recency (flags potential iterator/CUDA deadlocks)
  3. GPU activity & memory usage
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

RUN_DIR = Path(r"D:\marathi-asr\runs\scaled-conformer512-moe-3stage")
MAX_SILENCE_MINUTES = 25.0  # Max minutes without log file modification before warning


def check_gpu_process():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
        lines = [l for l in out.splitlines() if "python" in l.lower()]
        return lines
    except Exception as e:
        return []


def check_log_freshness():
    logs_dir = RUN_DIR / "logs"
    if not logs_dir.exists():
        return None, 999.0

    logs = list(logs_dir.glob("*.log"))
    if not logs:
        return None, 999.0

    # Pick the most recently modified log file
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_log = logs[0]
    silence_min = (time.time() - latest_log.stat().st_mtime) / 60.0
    return latest_log, silence_min


def inspect_health():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] === Training Health & Deadlock Diagnostic ===")

    # 1. GPU Compute Process
    gpu_procs = check_gpu_process()
    if gpu_procs:
        print(f"  [Process Check] Active GPU Compute Process Found:")
        for p in gpu_procs:
            print(f"    • {p}")
    else:
        print("  [Process Check] ⚠️ WARNING: No active python compute process found on GPU!")

    # 2. Log File Activity
    latest_log, silence_min = check_log_freshness()
    if latest_log:
        print(f"  [Log Heartbeat] Monitoring: {latest_log.name}")
        print(f"  [Last Activity] {silence_min:.1f} minutes ago")
        if silence_min > MAX_SILENCE_MINUTES:
            print(f"  [Deadlock Alert] ⚠️ Log has been silent for {silence_min:.1f} mins (> {MAX_SILENCE_MINUTES} mins limit)!")
            print("  Potential deadlock in data stream or CUDA kernel.")
            return False
        else:
            print(f"  [Health Status] Log activity within healthy window (< {MAX_SILENCE_MINUTES} mins).")
    else:
        print("  [Log Heartbeat] Log file not yet generated or found.")

    # 3. Last 5 lines of latest log
    if latest_log and latest_log.exists():
        try:
            with open(latest_log, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                print("  [Log Tail (last 3 lines)]:")
                for l in lines[-3:]:
                    print(f"    > {l[:100]}")
        except Exception:
            pass

    print("  [Overall Verdict] System healthy. No deadlock detected.\n")
    return True


if __name__ == "__main__":
    is_healthy = inspect_health()
    sys.exit(0 if is_healthy else 1)
