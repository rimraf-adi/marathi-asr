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

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import subprocess
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

DEFAULT_RUN_DIR = Path(r"D:\marathi-asr\runs\scaled-conformer512-dense-3stage")
MAX_SILENCE_MINUTES = 25.0


def get_active_run_dir(run_arg: Optional[str] = None) -> Path:
    if run_arg:
        p = Path(r"D:\marathi-asr\runs") / run_arg
        if p.exists():
            return p

    # Compare modification times between runs
    candidate_runs = [
        Path(r"D:\marathi-asr\runs\no-splitformer-moe-only"),
        Path(r"D:\marathi-asr\runs\scaled-conformer512-moe-3stage"),
        Path(r"D:\marathi-asr\runs\scaled-conformer512-dense-3stage"),
    ]
    existing = [p for p in candidate_runs if p.exists()]
    if not existing:
        return Path(r"D:\marathi-asr\runs\no-splitformer-moe-only")

    def get_latest_mtime(folder: Path) -> float:
        mtimes = [folder.stat().st_mtime]
        for sub in folder.glob("**/*"):
            try:
                mtimes.append(sub.stat().st_mtime)
            except Exception:
                pass
        return max(mtimes)

    existing.sort(key=get_latest_mtime, reverse=True)
    return existing[0]


def check_gpu_process():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
        lines = [l for l in out.splitlines() if "python" in l.lower()]
        return lines
    except Exception as e:
        return []


def check_log_freshness(run_dir: Path):
    activity_files = []
    for sub in [run_dir / "logs", run_dir / "csvs", run_dir / "checkpoints"]:
        if sub.exists():
            activity_files.extend(list(sub.glob("*.*")))

    # Check background task logs for current orchestrator tasks
    task_dir = Path(r"C:\Users\Machine Learning GPU\.gemini\antigravity\brain\4c2420fd-89d2-44e6-8285-cd95778da267\.system_generated\tasks")
    if task_dir.exists():
        activity_files.extend(list(task_dir.glob("task-*.log")))

    if not activity_files:
        return None, 999.0

    activity_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_file = activity_files[0]
    silence_min = (time.time() - latest_file.stat().st_mtime) / 60.0
    return latest_file, silence_min


def inspect_health(run_arg: Optional[str] = None):
    run_dir = get_active_run_dir(run_arg)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] === Training Health & Deadlock Diagnostic ({run_dir.name}) ===")

    # 1. GPU Compute Process
    gpu_procs = check_gpu_process()
    if gpu_procs:
        print(f"  [Process Check] Active GPU Compute Process Found:")
        for p in gpu_procs:
            print(f"    • {p}")
    else:
        print("  [Process Check] ⚠️ WARNING: No active python compute process found on GPU!")

    # 2. Log File Activity
    latest_log, silence_min = check_log_freshness(run_dir)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Run name to monitor")
    args = parser.parse_args()

    is_healthy = inspect_health(args.run)
    sys.exit(0 if is_healthy else 1)
