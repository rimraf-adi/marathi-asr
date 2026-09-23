"""
Sequential Multi-Stage Pipeline Orchestrator for Marathi ASR (3-Stage Training).
Executes Stages 1 through 4 sequentially based on a JSON config file.
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PYTHON_CMD = [sys.executable, "-u"]


def update_status(status_file: Path, stage_name: str, status: str, details: dict = None):
    data = {}
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data[stage_name] = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        **(details or {}),
    }

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_stage_checkpoint(ckpt_dir: Path, stage_name: str) -> Optional[Path]:
    stage_dir = ckpt_dir / stage_name
    if not stage_dir.exists():
        return None

    best = stage_dir / "best_model.pt"
    if best.exists():
        return best

    pts = sorted(stage_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]

    return None


def run_command_with_streaming_log(cmd: List[str], log_path: Path, stage_name: str, status_file: Path):
    print(f"\n{'='*70}")
    print(f"[Pipeline] Starting {stage_name.upper()}")
    print(f"[Command ] {' '.join(cmd)}")
    print(f"[Log File] {log_path.resolve()}")
    print(f"{'='*70}\n")

    update_status(status_file, stage_name, "running", {"cmd": cmd, "log_file": str(log_path)})
    t0 = time.time()

    with open(log_path, "a", encoding="utf-8", errors="replace") as f_log:
        f_log.write(f"\n--- Launching {stage_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f_log.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_log.write(line)
            f_log.flush()

        process.wait()
        retcode = process.returncode

    elapsed_min = (time.time() - t0) / 60.0
    if retcode != 0:
        update_status(status_file, stage_name, "failed", {"exit_code": retcode, "elapsed_min": round(elapsed_min, 1)})
        raise RuntimeError(f"Stage '{stage_name}' exited with error code {retcode} after {elapsed_min:.1f} mins!")

    update_status(status_file, stage_name, "completed", {"exit_code": 0, "elapsed_min": round(elapsed_min, 1)})
    print(f"\n[Pipeline] {stage_name.upper()} completed successfully in {elapsed_min:.1f} minutes.\n")


def main():
    parser = argparse.ArgumentParser(description="Marathi ASR 3-Stage Config-Based Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    # Validate config
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("Config must specify 'run_name'")

    # Setup directories
    run_dir = Path("runs") / run_name
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    status_file = run_dir / "pipeline_status.json"

    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Backup the config file into the run directory
    backup_config_path = run_dir / "config.json"
    shutil.copy2(config_path, backup_config_path)
    print(f"[Pipeline] Initialized run '{run_name}' in {run_dir.resolve()}")
    print(f"[Pipeline] Configuration backed up to {backup_config_path.resolve()}")

    start_stage = config.get("start_stage", 1)
    end_stage = config.get("end_stage", 4)

    # -------------------------------------------------------------
    # Stage 1: Pretraining
    # -------------------------------------------------------------
    if start_stage <= 1 <= end_stage:
        cfg = config.get("pretraining", {})
        s1_ckpt = find_stage_checkpoint(ckpt_dir, "stage1_pretrain")
        s1_cmd = [
            *PYTHON_CMD, "pretraining/train.py",
            "--run_dir", str(run_dir),
            "--epochs", str(cfg.get("epochs", 5)),
            "--batch_size", str(cfg.get("batch_size", 16)),
            "--lr", str(cfg.get("lr", 3e-4)),
            "--exp_name", "stage1_pretrain",
            "--stage", "pretrain",
            "--checkpoint_dir", str(ckpt_dir / "stage1_pretrain"),
            "--log_every", str(cfg.get("log_every", 25)),
            "--plot_every", str(cfg.get("plot_every", 100)),
            "--save_every", str(cfg.get("save_every", 1000)),
        ]
        if cfg.get("steps") is not None:
            s1_cmd.extend(["--steps", str(cfg.get("steps"))])
        if cfg.get("cache_dir") is not None:
            s1_cmd.extend(["--cache_dir", str(cfg.get("cache_dir"))])
        if cfg.get("cache_chunk_size") is not None:
            s1_cmd.extend(["--cache_chunk_size", str(cfg.get("cache_chunk_size"))])
        if cfg.get("max_cached_chunks") is not None:
            s1_cmd.extend(["--max_cached_chunks", str(cfg.get("max_cached_chunks"))])
        if s1_ckpt:
            s1_cmd.extend(["--resume", str(s1_ckpt)])
        run_command_with_streaming_log(s1_cmd, logs_dir / "stage1_pretrain.log", "stage1_pretrain", status_file)

    # -------------------------------------------------------------
    # Stage 2: 3-Dialect MoE Adaptation
    # -------------------------------------------------------------
    if start_stage <= 2 <= end_stage:
        cfg = config.get("moe", {})
        s1_ckpt = find_stage_checkpoint(ckpt_dir, "stage1_pretrain")
        if not s1_ckpt:
            raise FileNotFoundError("Could not find Stage 1 checkpoint.")

        s2_resume = find_stage_checkpoint(ckpt_dir, "stage2_moe")
        s2_cmd = [
            *PYTHON_CMD, "moe/train_moe.py",
            "--run_dir", str(run_dir),
            "--pretrained_ckpt", str(s1_ckpt),
            "--steps", str(cfg.get("steps", 25000)),
            "--batch_size", str(cfg.get("batch_size", 24)),
            "--grad_accum_steps", str(cfg.get("grad_accum_steps", 2)),
            "--lr", str(cfg.get("lr", 1e-4)),
            "--phase4a_steps", str(cfg.get("phase4a_steps", 7500)),
            "--checkpoint_dir", str(ckpt_dir / "stage2_moe"),
            "--stage_name", "stage2_moe",
            "--log_every", str(cfg.get("log_every", 25)),
            "--save_every", str(cfg.get("save_every", 1000)),
        ]
        if s2_resume:
            s2_cmd.extend(["--resume", str(s2_resume)])
        run_command_with_streaming_log(s2_cmd, logs_dir / "stage2_moe.log", "stage2_moe", status_file)

    # -------------------------------------------------------------
    # Stage 3: Final Alignment
    # -------------------------------------------------------------
    if start_stage <= 3 <= end_stage:
        cfg = config.get("final_alignment", {})
        s2_ckpt = find_stage_checkpoint(ckpt_dir, "stage2_moe")
        if not s2_ckpt:
            raise FileNotFoundError("Could not find Stage 2 checkpoint.")

        s3_resume = find_stage_checkpoint(ckpt_dir, "stage3_final_alignment")
        s3_cmd = [
            *PYTHON_CMD, "final_alignment/train_alignment.py",
            "--run_dir", str(run_dir),
            "--moe_ckpt", str(s2_ckpt),
            "--steps", str(cfg.get("steps", 8000)),
            "--batch_size", str(cfg.get("batch_size", 24)),
            "--grad_accum_steps", str(cfg.get("grad_accum_steps", 2)),
            "--lr", str(cfg.get("lr", 3e-5)),
            "--min_lr", str(cfg.get("min_lr", 5e-6)),
            "--checkpoint_dir", str(ckpt_dir / "stage3_final_alignment"),
            "--stage_name", "stage3_final_alignment",
            "--log_every", str(cfg.get("log_every", 25)),
            "--save_every", str(cfg.get("save_every", 500)),
        ]
        if s3_resume:
            s3_cmd.extend(["--resume", str(s3_resume)])
        run_command_with_streaming_log(s3_cmd, logs_dir / "stage3_final_alignment.log", "stage3_final_alignment", status_file)

    # -------------------------------------------------------------
    # Stage 4: Final Frozen Benchmark Evaluation
    # -------------------------------------------------------------
    if start_stage <= 4 <= end_stage:
        cfg = config.get("evaluation", {})
        final_ckpt = find_stage_checkpoint(ckpt_dir, "stage3_final_alignment")
        if not final_ckpt:
            raise FileNotFoundError("Could not find Stage 3 checkpoint.")

        eval_cmd = [
            *PYTHON_CMD, "eval_final_benchmark.py",
            "--checkpoint", str(final_ckpt),
            "--output_dir", str(run_dir / "eval_results"),
            "--tag", f"{run_name}_eval",
            "--chunk_size", "-1",  # No chunking / bidirectional
            "--beam_search",
            "--beam_width", str(cfg.get("beam_width", 16)),
        ]
        run_command_with_streaming_log(eval_cmd, logs_dir / "final_benchmark.log", "final_benchmark", status_file)

    print("\n[ALL COMPLETE] Full sequential pipeline finished successfully!")


if __name__ == "__main__":
    main()
