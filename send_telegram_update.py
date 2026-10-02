"""Root bridge to scripts/send_telegram_update.py for backward compatibility."""
import runpy
import sys
from pathlib import Path

target = Path(__file__).parent / "scripts" / "send_telegram_update.py"
if __name__ == "__main__":
    runpy.run_path(str(target), run_name="__main__")
