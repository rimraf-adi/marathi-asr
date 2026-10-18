import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch


class RunMetricsLogger:
    """
    Detailed Logging Module for ASR Pipeline telemetry.
    Supports CSV writing, JSONL streaming, Checkpoint management, and Terminal logging.
    """

    def __init__(
        self,
        stage_name: str,
        run_dir: str = "run3",
        total_steps: int = 10000,
        plot_interval: int = 100,
    ):
        self.stage_name = stage_name
        self.run_dir = Path(run_dir)
        self.total_steps = total_steps
        self.plot_interval = plot_interval

        # Setup telemetry directories
        self.ckpt_dir = self.run_dir / "checkpoints" / stage_name
        self.csv_path = self.run_dir / "csvs" / f"{stage_name}_metrics.csv"
        self.jsonl_path = self.run_dir / "logs" / f"{stage_name}_telemetry.jsonl"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self.csv_headers_written = False
        self.best_metric = None

        print(f"[{stage_name}] Telemetry initialized.")
        print(f"  - Checkpoints: {self.ckpt_dir.resolve()}")
        print(f"  - CSV Logging: {self.csv_path.resolve()}")
        print(f"  - JSONL Trace: {self.jsonl_path.resolve()}")

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """
        Logs a single training step's metrics to CSV and JSONL.
        """
        metrics_dict = {"step": step, "timestamp": time.time(), **metrics}

        # 1. JSONL Append (Detailed telemetry trace)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_dict) + "\n")

        # 2. CSV Write (For plotting & analysis)
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        if not self.csv_headers_written and file_exists:
            self.csv_headers_written = True

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_dict.keys()))
            if not self.csv_headers_written:
                writer.writeheader()
                self.csv_headers_written = True
            writer.writerow(metrics_dict)

    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metric_val: float,
        metric_name: str = "cer",
        lower_is_better: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        keep_last_k: int = 3,
        milestone_step: int = 0,
    ):
        """
        Saves model checkpoints, keeping track of the best performing model,
        and strictly rotating rolling checkpoints to keep the last 3 only.
        """
        ckpt_path = self.ckpt_dir / f"{self.stage_name}_step_{step}.pt"

        # Safely extract unwrapped model if using DDP/DataParallel
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        state_dict = {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            metric_name: metric_val,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(metadata or {}),
        }

        # Save current checkpoint
        torch.save(state_dict, ckpt_path)

        # Check if it's the best model so far
        is_best = False
        if self.best_metric is None:
            is_best = True
        else:
            if lower_is_better:
                if metric_val < self.best_metric:
                    is_best = True
            else:
                if metric_val > self.best_metric:
                    is_best = True

        if is_best:
            self.best_metric = metric_val
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(state_dict, best_path)
            print(f"  🌟 New Best Model! {metric_name} = {metric_val:.4f} (Saved to best_model.pt)")

        # Checkpoint rotation to prevent disk exhaustion:
        # Keep best_model.pt, milestones (step % milestone_step == 0), and latest keep_last_k checkpoints
        if keep_last_k is not None and keep_last_k > 0:
            existing_ckpts = []
            for p in self.ckpt_dir.glob(f"{self.stage_name}_step_*.pt"):
                stem_suffix = p.stem.split("_step_")[-1]
                if stem_suffix.isdigit():
                    existing_ckpts.append((int(stem_suffix), p))

            existing_ckpts.sort(key=lambda x: x[0])

            # Non-milestones can be rotated
            non_milestones = [
                (s, p) for s, p in existing_ckpts
                if milestone_step <= 0 or (s % milestone_step != 0)
            ]

            if len(non_milestones) > keep_last_k:
                to_prune = non_milestones[:-keep_last_k]
                for s, p in to_prune:
                    try:
                        p.unlink()
                        print(f"  [Checkpoint Rotation] Pruned older intermediate checkpoint: {p.name}")
                    except Exception as e:
                        print(f"  [Checkpoint Rotation] Warning: could not delete {p.name}: {e}")

