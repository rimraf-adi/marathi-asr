"""
Safe Intermediate Checkpoint Pruning Utility.
Preserves:
  1. ALL 'best_model.pt' files across every stage and run.
  2. ALL final terminal step checkpoints for each completed phase (e.g. step_350000, step_80000, step_18141).
  3. ALL files in the actively training run ('scaled-conformer512-dense-3stage').
Removes:
  - Redundant intermediate step checkpoints from completed historical runs.
"""

import os
import re
import shutil
from pathlib import Path

RUNS_DIR = Path(r"D:\marathi-asr\runs")
ACTIVE_RUN_NAME = "scaled-conformer512-dense-3stage"

def get_step_from_filename(filename: str):
    m = re.search(r"_step_(\d+)\.pt$", filename)
    if m:
        return int(m.group(1))
    return None

def prune_checkpoints(dry_run: bool = False):
    total_reclaimed_bytes = 0
    files_to_delete = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name == ACTIVE_RUN_NAME:
            continue

        for ckpt_dir in run_dir.glob("checkpoints/*"):
            if not ckpt_dir.is_dir():
                continue

            pt_files = list(ckpt_dir.glob("*.pt"))
            if not pt_files:
                continue

            # Separate best_model from step checkpoints
            step_files = []
            for p in pt_files:
                if p.name == "best_model.pt":
                    continue
                step = get_step_from_filename(p.name)
                if step is not None:
                    step_files.append((step, p))

            # If multiple step files exist, keep only the highest step (final milestone)
            if len(step_files) > 1:
                step_files.sort(key=lambda x: x[0], reverse=True)
                final_step_file = step_files[0][1]
                intermediate_files = [x[1] for x in step_files[1:]]

                for p in intermediate_files:
                    size = p.stat().st_size
                    files_to_delete.append((p, size))
                    total_reclaimed_bytes += size

    print(f"\n[Pruning Plan] Found {len(files_to_delete)} intermediate checkpoints ({total_reclaimed_bytes / (1024**3):.2f} GB to reclaim):")
    for p, sz in files_to_delete:
        print(f"  - {p.relative_to(RUNS_DIR)} ({sz / (1024**2):.1f} MB)")

    if dry_run:
        print("\n[Dry Run] No files were deleted.")
        return total_reclaimed_bytes

    deleted_count = 0
    for p, _ in files_to_delete:
        try:
            p.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete {p}: {e}")

    free_gb = shutil.disk_usage("D:\\").free / (1024**3)
    print(f"\n[Pruning Complete] Deleted {deleted_count} files.")
    print(f"Reclaimed: {total_reclaimed_bytes / (1024**3):.2f} GB.")
    print(f"Current Free Space on D: -> {free_gb:.2f} GB\n")
    return total_reclaimed_bytes

if __name__ == "__main__":
    prune_checkpoints(dry_run=False)
