"""
Standing Telegram Telemetry Dispatcher Daemon.
Runs periodically (configurable interval, default: every 60 minutes) in the background.
Collects real-time training telemetry, GPU health, checkpoint progress, and ETA,
then sends rich updates to the configured Telegram bot.
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

PYTHON_EXE = Path(r"D:\marathi-asr\.venv311\Scripts\python.exe")
UPDATE_SCRIPT = Path(r"D:\marathi-asr\scripts\send_telegram_update.py")


def run_telegram_update():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Executing periodic Telegram telemetry update...")
    try:
        res = subprocess.run(
            [str(PYTHON_EXE), str(UPDATE_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False
        )
        print(f"[{now_str}] Return code: {res.returncode}")
        if res.stdout:
            print(f"[{now_str}] Output: {res.stdout.strip()}")
        if res.stderr:
            print(f"[{now_str}] Stderr: {res.stderr.strip()}")
    except Exception as e:
        print(f"[{now_str}] Error executing update script: {e}")


def main():
    parser = argparse.ArgumentParser(description="Standing Telegram Update Daemon")
    parser.add_argument("--interval_minutes", type=int, default=120, help="Interval between updates in minutes (default: 120)")
    parser.add_argument("--immediate", action="store_true", help="Send an update immediately upon starting")
    args = parser.parse_args()

    interval_sec = args.interval_minutes * 60
    print(f"[Telegram Daemon] Initialized with interval of {args.interval_minutes} minutes ({interval_sec} seconds).")

    if args.immediate:
        run_telegram_update()

    while True:
        try:
            print(f"[Telegram Daemon] Sleeping for {args.interval_minutes} minutes until next update...")
            time.sleep(interval_sec)
            run_telegram_update()
        except KeyboardInterrupt:
            print("\n[Telegram Daemon] Stopped by user.")
            break
        except Exception as e:
            print(f"[Telegram Daemon] Encountered unexpected error: {e}. Retrying in 60s...")
            time.sleep(60)


if __name__ == "__main__":
    main()
