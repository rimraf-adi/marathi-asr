"""Sequential queue for the reduced-budget training ablations (resumable; completed steps are skipped).

    .venv311\\Scripts\\python.exe -u scripts/run_ablation_queue.py            # run everything
    .venv311\\Scripts\\python.exe -u scripts/run_ablation_queue.py --list     # print the plan only

Phase 0  : re-run Stage 1 self-supervised pretraining (MoE-architecture encoder, 1 epoch = 18,141 steps, ~30 h);
           the original Stage 1 checkpoints were deleted, and every variant must start from the SAME pretrained encoder.
Phase 1  : 20,000-step Stage 2 runs (warm-up 750, hard routing 3,750, learned routing the rest = 1/4 of the 80k recipe),
           ~6 h each, each followed by RESPIN + Kathbath evaluation:
             base_s0, base_s1, base_s2         (claim A: seed variance)
             scratch_s0                        (claim B: upcycled vs. randomly initialised experts)
             k1_s0, k2_s0, k4_s0               (claim C: number of experts; k=3 is base_s0)
             lower4_s0, all12_s0               (claim D: expert placement; base = blocks 5-12)
Results: evals/ablations/<name>.json, summary evals/ablation_training.json. Then run: python scripts/build_final_md.py
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv311" / "Scripts" / "python.exe")
RUN = ROOT / "runs" / "ablations"
S1_DIR = RUN / "stage1"
S1_CKPT = S1_DIR / "checkpoints" / "stage1_pretrain" / "best_model.pt"
STEPS, WARM, P4A = 20000, 750, 3750
BASE_LAYERS = "4,5,6,7,8,9,10,11"

VARIANTS = [  # name, num_experts, moe_layers, expert_init, seed, claim
    ("base_s0", 3, BASE_LAYERS, "upcycle", 0, "A,C,D reference"),
    ("base_s1", 3, BASE_LAYERS, "upcycle", 1, "A"),
    ("base_s2", 3, BASE_LAYERS, "upcycle", 2, "A"),
    ("scratch_s0", 3, BASE_LAYERS, "scratch", 0, "B"),
    ("k1_s0", 1, BASE_LAYERS, "upcycle", 0, "C"),
    ("k2_s0", 2, BASE_LAYERS, "upcycle", 0, "C"),
    ("k4_s0", 4, BASE_LAYERS, "upcycle", 0, "C"),
    ("lower4_s0", 3, "0,1,2,3", "upcycle", 0, "D"),
    ("all12_s0", 3, "0,1,2,3,4,5,6,7,8,9,10,11", "upcycle", 0, "D"),
]


KEEP_LAST = 3        # rolling checkpoints: newest N step files are kept, older ones are deleted
SAVE_EVERY = "500"   # about 9 minutes of training between saves, so a shutdown loses little work


def step_files(d):
    d = Path(d)
    if not d.exists():
        return []
    return sorted(d.glob("*_step_*.pt"), key=lambda f: int(f.stem.rsplit("_", 1)[-1]))


def prune(d, keep=KEEP_LAST):
    """Delete all but the newest `keep` step checkpoints (best_model.pt is never touched)."""
    for f in step_files(d)[:-keep]:
        try:
            f.unlink()
        except OSError:
            pass


def latest_step_ckpt(d):
    """Newest step checkpoint that loads cleanly (a file half-written during a shutdown is skipped), or None."""
    import torch
    for f in reversed(step_files(d)):
        try:
            torch.load(f, map_location="cpu", weights_only=False)
            return f
        except Exception as e:
            print(f"SKIP unreadable checkpoint {f.name}: {type(e).__name__}", flush=True)
    return None


def run(cmd, log, prune_dir=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        while True:
            try:
                p.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                if prune_dir:
                    prune(prune_dir)
    if prune_dir:
        prune(prune_dir)
    if p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("phase0: stage1 pretraining ->", S1_CKPT)
        for v in VARIANTS:
            print(v)
        return

    # ---- Phase 0
    s1_done = S1_DIR / "DONE"
    if not s1_done.exists():
        s1_resume = latest_step_ckpt(S1_DIR / "checkpoints" / "stage1_pretrain")
        run([PY, "-u", "pretraining/train.py", "--run_dir", str(S1_DIR), "--epochs", "1", "--steps", "18141",
             "--batch_size", "64", "--lr", "4e-4", "--exp_name", "stage1_pretrain", "--stage", "pretrain",
             "--checkpoint_dir", str(S1_DIR / "checkpoints" / "stage1_pretrain"),
             "--d_model", "512", "--n_heads", "8", "--ffn_expansion", "4",
             "--log_every", "25", "--plot_every", "100", "--save_every", SAVE_EVERY,
             "--cache_dir", "data_cache/pretrain_scaled", "--cache_chunk_size", "10000", "--max_cached_chunks", "10",
             "--no_indicvoices", "--no_indicvoices_r"] + (["--resume", str(s1_resume)] if s1_resume else []), S1_DIR / "stage1.log",
            prune_dir=S1_DIR / "checkpoints" / "stage1_pretrain")
        s1_done.write_text("ok")
    assert S1_CKPT.exists(), "stage 1 checkpoint missing"

    # ---- Phase 1
    summary = {}
    for name, k, layers, init, seed, claim in VARIANTS:
        out_json = ROOT / "evals" / "ablations" / f"{name}.json"
        ckpt = RUN / name / "checkpoints" / "stage2_moe" / "best_model.pt"
        done = RUN / name / "train.done"
        if not done.exists():
            resume = latest_step_ckpt(RUN / name / "checkpoints" / "stage2_moe")
            run([PY, "-u", "moe/train_moe.py", "--run_dir", str(RUN / name), "--pretrained_ckpt", str(S1_CKPT),
                 "--steps", str(STEPS), "--batch_size", "32", "--grad_accum_steps", "4", "--lr", "1.5e-4",
                 "--warmup_steps", "1000", "--phase4a_steps", str(P4A), "--ctc_warmup_steps", str(WARM),
                 "--checkpoint_dir", str(RUN / name / "checkpoints" / "stage2_moe"), "--stage_name", "stage2_moe",
                 "--log_every", "25", "--save_every", SAVE_EVERY,
                 "--num_experts", str(k), "--moe_layers", layers, "--expert_init", init, "--seed", str(seed)]
                + (["--resume", str(resume)] if resume else []), RUN / name / "train.log",
                prune_dir=RUN / name / "checkpoints" / "stage2_moe")
            done.write_text("ok")
        if not out_json.exists():
            run([PY, "-u", "scripts/eval_variant.py", "--name", name, "--ckpt", str(ckpt.relative_to(ROOT)),
                 "--num_experts", str(k), "--moe_layers", layers], RUN / name / "eval.log")
        summary[name] = dict(json.loads(out_json.read_text(encoding="utf-8")), claim=claim, seed=seed, expert_init=init)
        (ROOT / "evals" / "ablation_training.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("ALL DONE")


if __name__ == "__main__":
    main()
