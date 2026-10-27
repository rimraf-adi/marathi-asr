"""
Dialect-Specific LoRA Adapter Training & Benchmark Evaluation for Conformer ASR.
Trains 3 distinct Low-Rank Adapters (LoRA) on top of the frozen 130.8M Dense Conformer:
  - LoRA 0: D1 (Malvani / Konkan)
  - LoRA 1: D2 (Ahirani / Khandesh)
  - LoRA 2: D4 (Varhadi / Vidarbha)
Evaluates under:
  1. Oracle Dialect Routing (matched adapter selection per utterance)
  2. Cross-Dialect LoRA Swap Matrix (evaluating mismatched adapters)
  3. Direct head-to-head comparison vs. Autonomous Sparse MoE Conformer
Outputs structured results to runs/dialect_lora_adapters/
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import jiwer

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer, get_lr_scheduler
from moe.respin_dataset import build_or_load_speaker_split, LocalRESPINPrefetchLoader
from scripts.eval_final_benchmark import load_model_for_eval
from model.lora import inject_dialect_lora, set_active_lora_adapter, get_lora_state_dict

OUTPUT_DIR = Path("runs/dialect_lora_adapters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR = r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr"
META_PATH = Path(TEST_DIR) / "meta_test_mr.json"


def load_test_metadata() -> List[Dict[str, Any]]:
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = []
    for uid, info in meta.items():
        wav_path = Path(TEST_DIR) / info["wav_path"]
        samples.append({
            "uid": uid,
            "wav_path": str(wav_path),
            "text": sanitize_transcript(info.get("text", "").strip()),
            "dialect": info.get("dialect", "D3"),
            "speaker_id": info.get("speaker_id", "unknown"),
            "duration": float(info.get("duration", 0.0)),
        })
    return samples


def evaluate_lora_model(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    tokenizer: MarathiTokenizer,
    device: torch.device,
    adapter_mode: str = "oracle",  # 'oracle', 'none', or int (0, 1, 2)
) -> Dict[str, Any]:
    """Evaluates model with specified LoRA adapter mode."""
    dialect_stats = {
        d: {"err_words": 0.0, "tot_words": 0, "err_chars": 0.0, "tot_chars": 0}
        for d in ["D1", "D2", "D3", "D4", "ALL"]
    }

    dialect_to_adapter = {"D1": 0, "D2": 1, "D4": 2, "D3": None}

    model.eval()
    with torch.no_grad():
        for i, item in enumerate(samples):
            try:
                audio, sr = sf.read(item["wav_path"], dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    continue
                audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
            except Exception:
                continue

            d = item["dialect"]
            if adapter_mode == "oracle":
                active_idx = dialect_to_adapter.get(d, None)
            elif adapter_mode == "none":
                active_idx = None
            else:
                active_idx = int(adapter_mode)

            set_active_lora_adapter(model, active_idx)
            ctc_dict = model.forward_ctc(audio_t, chunk_size=None)
            logits = ctc_dict["log_probs"][0]  # (T, vocab_size)
            pred_tokens = logits.argmax(dim=-1).cpu().tolist()
            pred_text = sanitize_transcript(tokenizer.ctc_decode(pred_tokens))

            ref_text = item["text"]
            ref_words = len(ref_text.split())
            ref_chars = len(ref_text.replace(" ", ""))

            cer = compute_cer(ref_text, pred_text)
            try:
                wer = jiwer.wer(ref_text, pred_text) if ref_words > 0 else 0.0
            except Exception:
                wer = 1.0

            for group in [d, "ALL"]:
                dialect_stats[group]["err_words"] += wer * ref_words
                dialect_stats[group]["tot_words"] += ref_words
                dialect_stats[group]["err_chars"] += cer * ref_chars
                dialect_stats[group]["tot_chars"] += ref_chars

    results = {}
    for group, s in dialect_stats.items():
        w = round((s["err_words"] / max(1, s["tot_words"])) * 100.0, 2)
        c = round((s["err_chars"] / max(1, s["tot_chars"])) * 100.0, 2)
        results[group] = {"wer": w, "cer": c}
    return results


def train_dialect_lora(
    steps: int = 6000,
    batch_size: int = 32,
    grad_accum: int = 2,
    lr: float = 3e-4,
    r: int = 16,
    lora_alpha: float = 32.0,
    log_interval: int = 25,
    save_interval: int = 1000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n================================================================================")
    print(f"[LoRA Training] Multi-Dialect Conformer LoRA Adapters on {device} ({torch.cuda.get_device_name(0)})")
    print(f"  Steps: {steps} | Batch Size: {batch_size} (effective: {batch_size * grad_accum}) | LR: {lr}")
    print(f"  LoRA Rank (r): {r} | Alpha: {lora_alpha} | Dialects: 3 (D1 Malvani, D2 Ahirani, D4 Varhadi)")
    print(f"================================================================================\n")

    tokenizer = MarathiTokenizer("data_utils/vocab.json")

    # 1. Load Pretrained Dense Conformer-512
    dense_ckpt = "runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt"
    print(f"[Model] Loading pretrained dense Conformer backbone from: {dense_ckpt}")
    model = load_model_for_eval(dense_ckpt, device)

    # 2. Inject LoRA adapters
    inject_dialect_lora(model, num_dialects=3, r=r, lora_alpha=lora_alpha, lora_dropout=0.05, target_layers="all")
    model.train()

    # 3. Load RESPIN Train Split
    train_utts, _ = build_or_load_speaker_split()
    d1_utts = [u for u in train_utts if u.get("dialect") == "D1"]
    d2_utts = [u for u in train_utts if u.get("dialect") == "D2"]
    d4_utts = [u for u in train_utts if u.get("dialect") == "D4"]

    print(f"[Dataset] Dialect Training Pools:")
    print(f"  D1 (Malvani / Konkan): {len(d1_utts):,} utterances")
    print(f"  D2 (Ahirani / Khandesh): {len(d2_utts):,} utterances")
    print(f"  D4 (Varhadi / Vidarbha): {len(d4_utts):,} utterances")

    loader_d1 = LocalRESPINPrefetchLoader(d1_utts, tokenizer, batch_size=batch_size, queue_size=4)
    loader_d2 = LocalRESPINPrefetchLoader(d2_utts, tokenizer, batch_size=batch_size, queue_size=4)
    loader_d4 = LocalRESPINPrefetchLoader(d4_utts, tokenizer, batch_size=batch_size, queue_size=4)
    loaders = [loader_d1, loader_d2, loader_d4]

    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=100, total_steps=steps)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # Metrics CSV tracking
    metrics_csv_path = OUTPUT_DIR / "lora_training_metrics.csv"
    with open(metrics_csv_path, "w", encoding="utf-8") as f:
        f.write("step,dialect,loss,grad_norm,lr,steps_per_min,elapsed_sec,eta_min\n")

    print(f"\n[Training] Starting multi-dialect LoRA training for {steps:,} steps...")
    print(f"  Logging interval: every {log_interval} steps | Checkpoint interval: every {save_interval} steps")
    print(f"  Metrics file: {metrics_csv_path}\n", flush=True)

    t0 = time.time()
    optimizer.zero_grad()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    running_loss = 0.0
    last_grad_norm = 0.0

    for step in range(1, steps + 1):
        # Round-robin dialect selection: 0 (D1), 1 (D2), 2 (D4)
        dialect_idx = (step - 1) % 3
        set_active_lora_adapter(model, dialect_idx)

        batch = loaders[dialect_idx].queue.get()
        audio = batch["audio"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            ctc_dict = model.forward_ctc(audio, chunk_size=None)
            log_probs = ctc_dict["log_probs"]  # (B, T, vocab_size)
            output_lengths = ctc_dict["output_lengths"]

            # CTCLoss expects (T, B, C)
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss_fn(log_probs_t, targets, output_lengths, target_lengths)
            loss_scaled = loss / grad_accum

        scaler.scale(loss_scaled).backward()
        running_loss += loss.item()

        if step % grad_accum == 0:
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            last_grad_norm = float(gnorm.item()) if hasattr(gnorm, "item") else float(gnorm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Detailed Logging
        if step % log_interval == 0 or step == steps:
            window = log_interval if step % log_interval == 0 else (step % log_interval)
            avg_loss = running_loss / max(1, window)
            running_loss = 0.0
            elapsed = time.time() - t0
            steps_per_sec = step / max(1e-5, elapsed)
            steps_per_min = steps_per_sec * 60.0
            remaining_steps = max(0, steps - step)
            eta_min = (remaining_steps / max(1e-5, steps_per_sec)) / 60.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            d_name = ["D1 (Malvani)", "D2 (Ahirani)", "D4 (Varhadi)"][dialect_idx]

            print(
                f"[Step {step:5d}/{steps:5d}] | Dialect: {d_name:14s} | "
                f"Loss: {avg_loss:6.4f} | GradNorm: {last_grad_norm:5.2f} | "
                f"LR: {current_lr:.2e} | Speed: {steps_per_min:4.1f} steps/min | "
                f"ETA: {eta_min:5.1f} min",
                flush=True
            )

            # Append to CSV
            with open(metrics_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{d_name},{avg_loss:.5f},{last_grad_norm:.4f},{current_lr:.3e},{steps_per_min:.2f},{elapsed:.1f},{eta_min:.2f}\n")

        # Periodic checkpointing
        if step % save_interval == 0:
            ckpt_file = OUTPUT_DIR / f"lora_step_{step}.pt"
            torch.save({
                "step": step,
                "lora_state_dict": get_lora_state_dict(model),
                "lora_config": {
                    "r": r,
                    "lora_alpha": lora_alpha,
                    "num_dialects": 3,
                    "dialect_mapping": {"D1": 0, "D2": 1, "D4": 2},
                },
                "base_model": dense_ckpt,
            }, ckpt_file)
            print(f"  >>> [Checkpoint] Saved intermediate adapter weights: {ckpt_file.name}", flush=True)

    # Stop loader threads
    for l in loaders:
        l._stop_signal = True

    # 4. Save Trained LoRA Checkpoint
    lora_ckpt_path = OUTPUT_DIR / "best_lora_adapters.pt"
    lora_state = get_lora_state_dict(model)
    torch.save({
        "step": steps,
        "lora_state_dict": lora_state,
        "lora_config": {
            "r": r,
            "lora_alpha": lora_alpha,
            "num_dialects": 3,
            "dialect_mapping": {"D1": 0, "D2": 1, "D4": 2},
            "target_layers": "all",
        },
        "base_model": dense_ckpt,
    }, lora_ckpt_path)
    print(f"\n[Checkpoint] Saved final trained LoRA adapters to: {lora_ckpt_path}", flush=True)

    # 5. Comprehensive Benchmark Evaluation
    print(f"\n================================================================================")
    print(f"[LoRA Evaluation] Evaluating on IISc RESPIN Held-Out Test Set (2,170 Utterances)")
    print(f"================================================================================\n")

    test_samples = load_test_metadata()

    # A. Dense Baseline (No LoRA)
    print("[Eval] Evaluating Dense Baseline (No LoRA)...")
    res_dense = evaluate_lora_model(model, test_samples, tokenizer, device, adapter_mode="none")
    print(f"  Dense Baseline -> D1: {res_dense['D1']['wer']}%, D2: {res_dense['D2']['wer']}%, D3: {res_dense['D3']['wer']}%, D4: {res_dense['D4']['wer']}%, ALL: {res_dense['ALL']['wer']}%")

    # B. Oracle Dialect LoRA (Matched routing per utterance)
    print("\n[Eval] Evaluating Oracle Dialect LoRA (Matched Adapter)...")
    res_oracle = evaluate_lora_model(model, test_samples, tokenizer, device, adapter_mode="oracle")
    print(f"  Oracle LoRA    -> D1: {res_oracle['D1']['wer']}%, D2: {res_oracle['D2']['wer']}%, D3: {res_oracle['D3']['wer']}%, D4: {res_oracle['D4']['wer']}%, ALL: {res_oracle['ALL']['wer']}%")

    # C. Cross-Dialect LoRA Swap Matrix
    print("\n[Eval] Evaluating Cross-Dialect LoRA Swap Matrix...")
    swap_matrix = {}
    matched_map = {"D1": 0, "D2": 1, "D4": 2}
    for dialect, matched_id in matched_map.items():
        sub_samples = [s for s in test_samples if s["dialect"] == dialect]
        swap_matrix[dialect] = {"matched_adapter": f"lora_{matched_id}", "eval_wers": {}, "degradations_vs_matched": {}}
        for ad_id in [0, 1, 2]:
            sub_res = evaluate_lora_model(model, sub_samples, tokenizer, device, adapter_mode=str(ad_id))
            swap_matrix[dialect]["eval_wers"][f"lora_{ad_id}"] = sub_res[dialect]["wer"]

        matched_wer = swap_matrix[dialect]["eval_wers"][f"lora_{matched_id}"]
        for ad_id in [0, 1, 2]:
            cur_wer = swap_matrix[dialect]["eval_wers"][f"lora_{ad_id}"]
            delta = round(cur_wer - matched_wer, 2)
            swap_matrix[dialect]["degradations_vs_matched"][f"lora_{ad_id}"] = f"+{delta}%" if delta > 0 else f"{delta}%"

    print("\n[LoRA Swap Matrix Summary]:")
    for d, s in swap_matrix.items():
        print(f"  {d}: Matched ({s['matched_adapter']}) = {s['eval_wers'][s['matched_adapter']]}% | Degradations: {s['degradations_vs_matched']}")

    # Save summary JSON
    summary = {
        "dense_baseline_wer": res_dense,
        "oracle_lora_wer": res_oracle,
        "lora_swap_matrix": swap_matrix,
        "lora_config": {"r": r, "alpha": lora_alpha, "params_per_adapter": 3932160},
    }
    with open(OUTPUT_DIR / "lora_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Completed] LoRA benchmark summary saved to: {OUTPUT_DIR / 'lora_benchmark_summary.json'}\n")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate dialect LoRA adapters")
    parser.add_argument("--steps", type=int, default=6000, help="Total training steps (e.g. 6000 = ~384k utterances)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per forward")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--log_interval", type=int, default=25, help="Step interval for printing and CSV logging")
    parser.add_argument("--save_interval", type=int, default=1000, help="Step interval for saving checkpoint")
    parser.add_argument("--clean", action="store_true", help="Clear previous checkpoints in output directory before starting")
    args = parser.parse_args()

    if args.clean and OUTPUT_DIR.exists():
        import shutil
        for item in OUTPUT_DIR.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"[Cleanup] Cleared all existing files and checkpoints from {OUTPUT_DIR}", flush=True)

    train_dialect_lora(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        r=args.rank,
        lora_alpha=args.alpha,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
