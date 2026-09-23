"""
Semantic Checkpoint Organizer for Marathi Streaming Conformer ASR
Maps run checkpoints into semantic stage directories:
  checkpoints/pretrain/
  checkpoints/causal/
  checkpoints/multi_exit/
  checkpoints/moe/
  checkpoints/calibrated/
"""

import os
import re
import shutil
import argparse
from pathlib import Path


STAGE_MAP = {
    "pretrain": "checkpoints/pretrain",
    "causal": "checkpoints/causal",
    "multi_exit": "checkpoints/multi_exit",
    "moe": "checkpoints/moe",
    "calibrated": "checkpoints/calibrated",
}


def organize_checkpoints(run_dir: str = "runs/marathi_conformer_production", stage: str = "pretrain", move: bool = False):
    dest_dir = Path(STAGE_MAP.get(stage, f"checkpoints/{stage}"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    src_ckpt_dir = Path(run_dir) / "checkpoints"
    if not src_ckpt_dir.exists():
        print(f"[Warn] Source directory not found: {src_ckpt_dir}")
        return

    pt_files = sorted(list(src_ckpt_dir.glob("*.pt")))
    print(f"[Organizer] Scanning {len(pt_files)} checkpoints in '{src_ckpt_dir}' for stage '{stage}' -> '{dest_dir}'")

    highest_step = -1
    highest_file = None

    for pt_file in pt_files:
        filename = pt_file.name
        match = re.search(r"step_?(\d+)", filename, re.IGNORECASE)
        if match:
            step_num = int(match.group(1))
            semantic_name = f"conformer_{stage}_step_{step_num:06d}.pt"
            dest_file = dest_dir / semantic_name

            if not dest_file.exists() or dest_file.stat().st_size != pt_file.stat().st_size:
                if move:
                    shutil.move(str(pt_file), str(dest_file))
                    print(f"  Moved: {filename} -> {dest_file}")
                else:
                    shutil.copy2(str(pt_file), str(dest_file))
                    print(f"  Copied: {filename} -> {dest_file}")
            else:
                print(f"  Already in place: {dest_file.name}")

            if step_num > highest_step:
                highest_step = step_num
                highest_file = dest_file

    # Point latest & final if applicable
    if highest_file and highest_file.exists():
        latest_file = dest_dir / f"conformer_{stage}_latest.pt"
        shutil.copy2(str(highest_file), str(latest_file))
        print(f"[Organizer] Updated '{latest_file.name}' -> (Step {highest_step})")

        if highest_step >= 10000:
            final_file = dest_dir / f"conformer_{stage}_final.pt"
            shutil.copy2(str(highest_file), str(final_file))
            print(f"[Organizer] Created '{final_file.name}' -> (Step {highest_step})")

    print(f"[Organizer] Semantic checkpoints organized successfully in '{dest_dir}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize checkpoints semantically by training stage")
    parser.add_argument("--run_dir", type=str, default="runs/marathi_conformer_production", help="Run directory containing checkpoints")
    parser.add_argument("--stage", type=str, default="pretrain", help="Target training stage (pretrain, causal, multi_exit, moe)")
    parser.add_argument("--move", action="store_true", help="Move instead of copy files")

    args = parser.parse_args()
    organize_checkpoints(run_dir=args.run_dir, stage=args.stage, move=args.move)
