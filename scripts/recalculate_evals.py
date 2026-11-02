"""
Live Benchmark Re-evaluation Engine.
Recalculates exact metrics (WER, CER, RTFx, 95% Bootstrap CIs) from scratch for all models on RESPIN test suite.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torchaudio
import soundfile as sf
import numpy as np
import jiwer
from tqdm import tqdm

from normalizer import normalize, NORM_VERSION
from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from moe.upcycling import load_moe_model
from model.asr_model import StreamingASRModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {DEVICE}: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def bootstrap_ci(errs: np.ndarray, lens: np.ndarray, n_boot: int = 1000, seed: int = 42) -> List[float]:
    rng = np.random.default_rng(seed)
    n = len(errs)
    if n == 0:
        return [0.0, 0.0]
    wers = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        wers.append(errs[idx].sum() / max(lens[idx].sum(), 1))
    return [
        round(float(np.percentile(wers, 2.5) * 100), 2),
        round(float(np.percentile(wers, 97.5) * 100), 2),
    ]

def load_respin_metadata(test_dir: str = r"D:\dialect-norm\IISc_RESPIN_test_mr\IISc_RESPIN_test_mr"):
    test_path = Path(test_dir)
    meta_path = test_path / "meta_test_mr.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = []
    for uid, info in meta.items():
        wav_file = test_path / info["wav_path"]
        samples.append({
            "uid": uid,
            "wav_path": str(wav_file),
            "text": info.get("text", "").strip(),
            "dialect": info.get("dialect", "D3"),
            "duration": float(info.get("duration", 0.0)),
            "speaker_id": info.get("speaker_id", "unknown"),
        })
    return samples

def compute_metrics(samples: List[Dict[str, Any]], preds: List[str]):
    by_dialect = {"D1": [], "D2": [], "D3": [], "D4": [], "ALL": []}
    for s, p in zip(samples, preds):
        ref_n = normalize(s["text"])
        hyp_n = normalize(p)
        item = {"ref": ref_n, "hyp": hyp_n, "raw_ref": s["text"], "raw_hyp": p}
        by_dialect[s["dialect"]].append(item)
        by_dialect["ALL"].append(item)

    metrics = {}
    for d, items in by_dialect.items():
        refs = [it["ref"] for it in items]
        hyps = [it["hyp"] for it in items]
        keep = [i for i, r in enumerate(refs) if r.strip()]
        r_k = [refs[i] for i in keep]
        h_k = [hyps[i] for i in keep]

        err_list, len_list = [], []
        for r, h in zip(r_k, h_k):
            o = jiwer.process_words(r, h)
            err_list.append(o.substitutions + o.deletions + o.insertions)
            len_list.append(len(r.split()))

        errs = np.array(err_list)
        lens = np.array(len_list)
        wer = round(float(errs.sum() / max(lens.sum(), 1) * 100), 2)
        cer = round(float(jiwer.cer(r_k, h_k) * 100), 2)
        ci = bootstrap_ci(errs, lens) if d == "ALL" else None
        metrics[d] = {"wer": wer, "cer": cer, "n": len(r_k), "ci95": ci}

    return metrics

def evaluate_moe_model(name: str, ckpt_path: str, samples: List[Dict[str, Any]], d_model: int = 512, n_heads: int = 8, ffn_expansion: int = 4, batch_size: int = 32):
    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    print(f"\n[{name}] Loading MoE checkpoint: {ckpt_path} (d_model={d_model}, n_heads={n_heads})...")
    model = load_moe_model(ckpt_path, device=DEVICE, d_model=d_model, n_heads=n_heads, ffn_expansion=ffn_expansion)
    model.eval()

    total_utts = len(samples)
    preds = []
    wall_time = 0.0
    tot_audio_sec = sum(s["duration"] for s in samples)

    print(f"[{name}] Transcribing {total_utts} utterances with batch_size={batch_size}...")
    for i in range(0, total_utts, batch_size):
        batch = samples[i : i + batch_size]
        audio_tensors = []
        for s in batch:
            arr, sr = sf.read(s["wav_path"], dtype="float32")
            if len(arr.shape) > 1:
                arr = arr.mean(axis=1)
            audio_tensors.append(torch.from_numpy(arr))

        lens = torch.tensor([len(a) for a in audio_tensors], dtype=torch.long)
        max_l = lens.max().item()
        padded = torch.zeros(len(batch), max_l, dtype=torch.float32, device=DEVICE)
        for idx, a in enumerate(audio_tensors):
            padded[idx, :len(a)] = a.to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            ctc_dict = model.forward_ctc(padded, chunk_size=None)
            logits = ctc_dict["log_probs"]
            for b in range(len(batch)):
                tokens = logits[b].argmax(dim=-1).cpu().tolist()
                text = sanitize_transcript(tokenizer.ctc_decode(tokens))
                preds.append(text)
        wall_time += time.perf_counter() - t0

    rtfx = tot_audio_sec / max(wall_time, 1e-5)
    return preds, rtfx, wall_time

def evaluate_dense_model(name: str, ckpt_path: str, samples: List[Dict[str, Any]], d_model: int = 512, n_heads: int = 8, ffn_expansion: int = 8, batch_size: int = 32):
    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    print(f"\n[{name}] Loading Dense checkpoint: {ckpt_path} (d_model={d_model}, n_heads={n_heads}, ffn={ffn_expansion})...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = StreamingASRModel(feat_dim=80, d_model=d_model, n_heads=n_heads, ffn_expansion=ffn_expansion, num_layers=12, vocab_size=105)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()

    total_utts = len(samples)
    preds = []
    wall_time = 0.0
    tot_audio_sec = sum(s["duration"] for s in samples)

    print(f"[{name}] Transcribing {total_utts} utterances with batch_size={batch_size}...")
    for i in range(0, total_utts, batch_size):
        batch = samples[i : i + batch_size]
        audio_tensors = []
        for s in batch:
            arr, sr = sf.read(s["wav_path"], dtype="float32")
            if len(arr.shape) > 1:
                arr = arr.mean(axis=1)
            audio_tensors.append(torch.from_numpy(arr))

        lens = torch.tensor([len(a) for a in audio_tensors], dtype=torch.long)
        max_l = lens.max().item()
        padded = torch.zeros(len(batch), max_l, dtype=torch.float32, device=DEVICE)
        for idx, a in enumerate(audio_tensors):
            padded[idx, :len(a)] = a.to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            ctc_dict = model.forward_ctc(padded, chunk_size=None)
            logits = ctc_dict["log_probs"]
            for b in range(len(batch)):
                tokens = logits[b].argmax(dim=-1).cpu().tolist()
                text = sanitize_transcript(tokenizer.ctc_decode(tokens))
                preds.append(text)
        wall_time += time.perf_counter() - t0

    rtfx = tot_audio_sec / max(wall_time, 1e-5)
    return preds, rtfx, wall_time

def evaluate_sravaani_live(samples: List[Dict[str, Any]], batch_size: int = 32):
    from transformers import AutoModel
    token = os.environ.get("HF_TOKEN")
    print("\n[SraVaani] Loading ARTPARK-IISc/SraVaani-1.0...")
    model = AutoModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0", trust_remote_code=True, token=token).to(DEVICE)
    model.eval()

    wav_paths = [s["wav_path"] for s in samples]
    tot_audio_sec = sum(s["duration"] for s in samples)
    t0 = time.perf_counter()
    hyps = model.transcribe(wav_paths, batch_size=batch_size, return_hypotheses=True)
    wall_time = time.perf_counter() - t0

    preds = []
    for h in hyps:
        raw_text = h.text if hasattr(h, "text") else str(h)
        preds.append(sanitize_transcript(raw_text))

    rtfx = tot_audio_sec / max(wall_time, 1e-5)
    return preds, rtfx, wall_time

def evaluate_indic_conformer_live(samples: List[Dict[str, Any]]):
    from transformers import AutoModel
    token = os.environ.get("HF_TOKEN")
    print("\n[IndicConformer-600M] Loading ai4bharat/indic-conformer-600m-multilingual...")
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True, token=token).to(DEVICE)
    model.eval()

    tot_audio_sec = sum(s["duration"] for s in samples)
    preds = []
    t0 = time.perf_counter()
    for s in tqdm(samples, desc="IndicConformer"):
        wav, sr = torchaudio.load(s["wav_path"])
        if wav.ndim > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
        pred = model(wav, "mr", "ctc")
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        preds.append(sanitize_transcript(str(pred)))
    wall_time = time.perf_counter() - t0

    rtfx = tot_audio_sec / max(wall_time, 1e-5)
    return preds, rtfx, wall_time

if __name__ == "__main__":
    samples = load_respin_metadata()
    print(f"Loaded {len(samples)} RESPIN test utterances.")
    all_results = {}

    # Model 1: 130M MoE
    ckpt_130m_moe = "runs/scaled-conformer512-moe-3stage/checkpoints/stage3_final_alignment/best_model.pt"
    if os.path.exists(ckpt_130m_moe):
        preds, rtfx, wall = evaluate_moe_model("130M MoE", ckpt_130m_moe, samples, d_model=512, n_heads=8, ffn_expansion=4)
        m = compute_metrics(samples, preds)
        all_results["marathi-conformer-130m-moe"] = {"metrics": m, "rtfx": round(rtfx, 1)}
        print(f"130M MoE Complete! ALL WER: {m['ALL']['wer']}%, CER: {m['ALL']['cer']}%, RTFx: {rtfx:.1f}x")

    # Model 2: 33M MoE
    ckpt_33m_moe = "runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt"
    if os.path.exists(ckpt_33m_moe):
        preds, rtfx, wall = evaluate_moe_model("33M MoE", ckpt_33m_moe, samples, d_model=256, n_heads=4, ffn_expansion=4)
        m = compute_metrics(samples, preds)
        all_results["marathi-conformer-33m-moe"] = {"metrics": m, "rtfx": round(rtfx, 1)}
        print(f"33M MoE Complete! ALL WER: {m['ALL']['wer']}%, CER: {m['ALL']['cer']}%, RTFx: {rtfx:.1f}x")

    # Model 3: 130M Dense
    ckpt_130m_dense = "runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt"
    if os.path.exists(ckpt_130m_dense):
        preds, rtfx, wall = evaluate_dense_model("130M Dense", ckpt_130m_dense, samples, d_model=512, n_heads=8, ffn_expansion=8)
        m = compute_metrics(samples, preds)
        all_results["marathi-conformer-130m-dense"] = {"metrics": m, "rtfx": round(rtfx, 1)}
        print(f"130M Dense Complete! ALL WER: {m['ALL']['wer']}%, CER: {m['ALL']['cer']}%, RTFx: {rtfx:.1f}x")

    # Model 4: 33M Dense
    ckpt_33m_dense = "runs/no-splitformer-moe-only/checkpoints/stage1_pretrain/best_model.pt"
    if os.path.exists(ckpt_33m_dense):
        preds, rtfx, wall = evaluate_dense_model("33M Dense", ckpt_33m_dense, samples, d_model=256, n_heads=4, ffn_expansion=4)
        m = compute_metrics(samples, preds)
        all_results["marathi-conformer-33m-dense"] = {"metrics": m, "rtfx": round(rtfx, 1)}
        print(f"33M Dense Complete! ALL WER: {m['ALL']['wer']}%, CER: {m['ALL']['cer']}%, RTFx: {rtfx:.1f}x")

    # Save fresh recalculated results
    out_file = "inference/results/respin/fresh_recalculated_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all fresh recalculated metrics to {out_file}!")
