"""
Master Vistaar Benchmark Evaluation Pipeline for Marathi ASR.
Supports complete 5-way model/decoding ablation:
  1. Our Model (Greedy CTC)
  2. Our Model (Prefix Beam Search - Lexicon)
  3. Our Model (Prefix Beam Search + KenLM 5-gram)
  4. ARTPARK-IISc / SraVaani 1.0 (CTC)
  5. AI4Bharat / Indic Conformer 600M (CTC)
Across all Vistaar Marathi benchmarks:
  - IndicTTS
  - Kathbath
  - Kathbath Hard (kathbath_noisy)
  - CommonVoice
  - FLEURS
  - MUCS

Generates:
  - Benchmark Summary CSV & Markdown Table
  - Per-Utterance Detailed CSV
  - Qualitative Analysis Report (Easy, Medium, Hard breakdown with linguistic root causes)
  - Master Comparison Table against IndicWhisper
Preserves all extracted audio datasets on disk (no purging).
"""

import os
import sys
import gc
import json
import time
import shutil
import csv
from pathlib import Path
from typing import Dict, Any, List

# Windows UTF-8 configuration
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(root_dir, ".env"))

import torch
import torchaudio
import soundfile as sf
import jiwer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.utils import compute_cer
from moe.upcycling import load_moe_model
from decoding.beam_search_decoder import MarathiBeamSearchDecoder, load_marathi_wordlist
from inference.pipeline import normalize_marathi_text, OurASRModel, SraVaaniModel, IndicConformerModel
from inference.vistaar_benchmark.fast_vistaar import extract_marathi_benchmark
from inference.vistaar_benchmark.eval_vistaar import load_vistaar_manifest, VistaarEvaluator
from inference.vistaar_benchmark.qualitative_analyzer import generate_qualitative_report

# Official Vistaar URLs
VISTAAR_URLS = {
    "indictts": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/indictts.zip",
    "kathbath": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath.zip",
    "kathbath_noisy": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath_noisy.zip",
    "commonvoice": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/commonvoice.zip",
    "fleurs": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/fleurs.zip",
    "mucs": "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/mucs.zip",
}

INDICWHISPER_WER = {
    "indictts": 11.4,
    "kathbath": 19.9,
    "kathbath_noisy": 22.1,
    "commonvoice": 22.8,
    "fleurs": 20.5,
    "mucs": 12.8,
}


def evaluate_benchmark(benchmark_name: str, manifest_path: Path, output_dir: Path, batch_size: int = 32, device: str = "cuda", run_external: bool = True):
    """Evaluates all models and decoding variants on the given manifest."""
    evaluator = VistaarEvaluator(output_dir=str(output_dir), device=device)
    samples = load_vistaar_manifest(str(manifest_path))

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {benchmark_name.upper()} ({len(samples)} utterances)")
    print(f"{'='*70}")

    summaries = []

    # 1. Load Our Model + Decoders
    print(f"\n[1/3] Loading Our Model (MoE Conformer) & Beam Decoders...")
    model_our = load_moe_model(evaluator.checkpoint_path, device=torch.device(device))
    model_our.eval()
    tokenizer = MarathiTokenizer(evaluator.vocab_path)
    words = load_marathi_wordlist(max_words=30000)

    print("[Decoder 1/4] Initializing Pure Greedy CTC...")
    print("[Decoder 2/4] Initializing Prefix Beam Search (Lexicon)...")
    dec_beam = MarathiBeamSearchDecoder(tokenizer, unigram_words=words, kenlm_path=None)
    print("[Decoder 3/4] Initializing Prefix Beam Search + KenLM Default (alpha=0.5, beta=1.5)...")
    dec_kenlm_def = MarathiBeamSearchDecoder(tokenizer, unigram_words=words, kenlm_path="lm/marathi_5gram.binary", alpha=0.5, beta=1.5)
    print("[Decoder 4/4] Initializing Prefix Beam Search + KenLM Soft (alpha=0.15, beta=1.0)...")
    dec_kenlm_soft = MarathiBeamSearchDecoder(tokenizer, unigram_words=words, kenlm_path="lm/marathi_5gram.binary", alpha=0.15, beta=1.0)

    preds_greedy = []
    preds_beam = []
    preds_kenlm_def = []
    preds_kenlm_soft = []

    t0 = time.time()
    for i in tqdm(range(0, len(samples), 16), desc="Our Model Forward"):
        batch = samples[i:i + 16]
        audios = []
        for item in batch:
            wav, sr = torchaudio.load(item["wav_path"])
            if wav.ndim > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
            audios.append(wav.squeeze(0))

        # Forward pass on GPU
        b_cur = len(audios)
        lens = torch.tensor([len(a) for a in audios], dtype=torch.long)
        max_len = lens.max().item()
        padded = torch.zeros(b_cur, max_len, dtype=torch.float32, device=device)
        for idx, a in enumerate(audios):
            padded[idx, :len(a)] = a.to(device)

        with torch.no_grad():
            ctc_dict = model_our.forward_ctc(padded, chunk_size=None)
            logits_batch = ctc_dict["log_probs"].cpu().float().numpy()

        for b in range(b_cur):
            l = logits_batch[b]
            # 1. Greedy
            tokens = l.argmax(axis=-1).tolist()
            preds_greedy.append(sanitize_transcript(tokenizer.ctc_decode(tokens)))
            # 2. Prefix Beam (Lexicon)
            preds_beam.append(dec_beam.decode(l, beam_width=16))
            # 3. KenLM Default
            preds_kenlm_def.append(dec_kenlm_def.decode(l, beam_width=16))
            # 4. KenLM Soft
            preds_kenlm_soft.append(dec_kenlm_soft.decode(l, beam_width=16))

    comp_time_our = time.time() - t0

    # Evaluate all 4 variants
    s_greedy, det_greedy = evaluator.evaluate_model("Our Model (Greedy CTC)", benchmark_name, samples, preds_greedy, comp_time_our)
    s_beam, det_beam = evaluator.evaluate_model("Our Model (Lexicon Beam)", benchmark_name, samples, preds_beam, comp_time_our)
    s_kdef, det_kdef = evaluator.evaluate_model("Our Model (KenLM Default α=0.5)", benchmark_name, samples, preds_kenlm_def, comp_time_our)
    s_ksoft, det_ksoft = evaluator.evaluate_model("Our Model (KenLM Soft α=0.15)", benchmark_name, samples, preds_kenlm_soft, comp_time_our)

    summaries.extend([s_greedy, s_beam, s_kdef, s_ksoft])

    del model_our
    torch.cuda.empty_cache()
    gc.collect()

    # 2. SraVaani 1.0
    preds_sra = []
    if run_external:
        cached_sra_csv = output_dir / f"{benchmark_name}_sravaani_detailed.csv"
        cached_all_csv = output_dir / f"{benchmark_name}_all_models_detailed.csv"
        loaded_sra = False
        if cached_sra_csv.exists():
            try:
                import pandas as pd
                df_sra = pd.read_csv(cached_sra_csv, encoding="utf-8")
                if len(df_sra) == len(samples) and "prediction" in df_sra.columns:
                    preds_sra = [str(p) if pd.notna(p) else "" for p in df_sra["prediction"]]
                    loaded_sra = True
                    print(f"\n[2/3] Reusing {len(preds_sra)} genuine SraVaani 1.0 predictions from {cached_sra_csv.name}")
            except Exception as e:
                print(f"[Warning] Could not load cached SraVaani: {e}")
        elif cached_all_csv.exists():
            try:
                import pandas as pd
                df_all = pd.read_csv(cached_all_csv, encoding="utf-8")
                if len(df_all) == len(samples) and "sra_pred" in df_all.columns and not df_all["sra_pred"].isna().all():
                    preds_sra = [str(p) if pd.notna(p) else "" for p in df_all["sra_pred"]]
                    loaded_sra = True
                    print(f"\n[2/3] Reusing {len(preds_sra)} genuine SraVaani 1.0 predictions from {cached_all_csv.name}")
            except Exception as e:
                pass

        if not loaded_sra:
            print(f"\n[2/3] Evaluating SraVaani 1.0 (ARTPARK-IISc)...")
            sravaani = SraVaaniModel(device=device)
            t0 = time.time()
            wav_paths = [s["wav_path"] for s in samples]
            for i in tqdm(range(0, len(wav_paths), batch_size), desc="SraVaani"):
                batch_paths = wav_paths[i:i + batch_size]
                preds_sra.extend(sravaani.transcribe_paths(batch_paths, batch_size=batch_size))
            comp_time_sra = time.time() - t0
            del sravaani
            torch.cuda.empty_cache()
            gc.collect()
        else:
            comp_time_sra = 1.0

        s_sra, det_sra = evaluator.evaluate_model("SraVaani 1.0 (ARTPARK-IISc)", benchmark_name, samples, preds_sra, comp_time_sra)
        summaries.append(s_sra)

    # 3. Indic Conformer 600M
    preds_ind = []
    if run_external:
        cached_ind_csv = output_dir / f"{benchmark_name}_indic_conformer_detailed.csv"
        cached_all_csv = output_dir / f"{benchmark_name}_all_models_detailed.csv"
        loaded_ind = False
        if cached_ind_csv.exists():
            try:
                import pandas as pd
                df_ind = pd.read_csv(cached_ind_csv, encoding="utf-8")
                if len(df_ind) == len(samples) and "prediction" in df_ind.columns:
                    preds_ind = [str(p) if pd.notna(p) else "" for p in df_ind["prediction"]]
                    loaded_ind = True
                    print(f"\n[3/3] Reusing {len(preds_ind)} genuine Indic Conformer predictions from {cached_ind_csv.name}")
            except Exception as e:
                print(f"[Warning] Could not load cached Indic Conformer: {e}")
        elif cached_all_csv.exists():
            try:
                import pandas as pd
                df_all = pd.read_csv(cached_all_csv, encoding="utf-8")
                if len(df_all) == len(samples) and "ind_pred" in df_all.columns and not df_all["ind_pred"].isna().all():
                    preds_ind = [str(p) if pd.notna(p) else "" for p in df_all["ind_pred"]]
                    loaded_ind = True
                    print(f"\n[3/3] Reusing {len(preds_ind)} genuine Indic Conformer predictions from {cached_all_csv.name}")
            except Exception as e:
                pass

        if not loaded_ind:
            print(f"\n[3/3] Evaluating Indic Conformer 600M (AI4Bharat)...")
            indic = IndicConformerModel(device=device)
            def _transcribe(s):
                try: return indic.transcribe_single(s["wav_path"])
                except Exception: return ""
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=12) as ex:
                preds_ind = list(tqdm(ex.map(_transcribe, samples), total=len(samples), desc="Indic Conformer"))
            comp_time_ind = time.time() - t0
            del indic
            torch.cuda.empty_cache()
            gc.collect()
        else:
            comp_time_ind = 1.0

        s_ind, det_ind = evaluator.evaluate_model("Indic Conformer 600M (AI4Bharat)", benchmark_name, samples, preds_ind, comp_time_ind)
        summaries.append(s_ind)

    # Save summary CSV & Markdown Table
    sum_csv = output_dir / f"{benchmark_name}_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    md_file = output_dir / f"{benchmark_name}_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Vistaar Benchmark Report: `{benchmark_name.upper()}` (Marathi)\n\n")
        f.write(f"- **Total Utterances**: {len(samples)} ({sum(s['duration'] for s in samples)/3600.0:.2f} hours)\n")
        f.write(f"- **Manifest**: `{manifest_path}`\n\n")
        f.write("| Model / Decoding Variant | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in summaries:
            f.write(f"| **{s['model_name']}** | **{s['cer_percent']:.2f}%** | **{s['wer_percent']:.2f}%** | {s['ser_percent']:.2f}% | {s['exact_match_acc']:.2f}% | {s['rtf']}x | {s['latency_ms_per_utt']:.1f} ms |\n")

    # Generate unified detailed per-utterance data for qualitative reporting
    samples_data = []
    for idx, s in enumerate(samples):
        row = {
            "uid": s["uid"],
            "wav_path": s["wav_path"],
            "reference": s["text"],
            "greedy_pred": preds_greedy[idx],
            "greedy_cer": det_greedy[idx]["norm_cer_percent"],
            "greedy_wer": det_greedy[idx]["norm_wer_percent"],
            "beam_pred": preds_beam[idx],
            "beam_cer": det_beam[idx]["norm_cer_percent"],
            "beam_wer": det_beam[idx]["norm_wer_percent"],
            "kenlm_def_pred": preds_kenlm_def[idx],
            "kenlm_def_cer": det_kdef[idx]["norm_cer_percent"],
            "kenlm_def_wer": det_kdef[idx]["norm_wer_percent"],
            "kenlm_soft_pred": preds_kenlm_soft[idx],
            "kenlm_soft_cer": det_ksoft[idx]["norm_cer_percent"],
            "kenlm_soft_wer": det_ksoft[idx]["norm_wer_percent"],
        }
        if preds_sra:
            row["sra_pred"] = preds_sra[idx]
            row["sra_wer"] = det_sra[idx]["norm_wer_percent"]
        if preds_ind:
            row["ind_pred"] = preds_ind[idx]
            row["ind_wer"] = det_ind[idx]["norm_wer_percent"]
        samples_data.append(row)

    # Save detailed CSV
    det_csv = output_dir / f"{benchmark_name}_all_models_detailed.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples_data[0].keys()))
        writer.writeheader()
        writer.writerows(samples_data)

    # Generate Qualitative Report
    qual_report = output_dir / f"{benchmark_name}_qualitative_report.md"
    generate_qualitative_report(benchmark_name, samples_data, qual_report, num_examples_per_bucket=5)

    print(f"\n[Artifacts Saved]")
    print(f"  Summary CSV:        {sum_csv}")
    print(f"  Benchmark Report:   {md_file}")
    print(f"  Detailed CSV:       {det_csv}")
    print(f"  Qualitative Report: {qual_report}")
    return summaries


def build_master_vistaar_report(output_dir: Path):
    """Compiles all benchmark summary CSVs into master tables comparing all models + IndicWhisper."""
    benchmarks = ["indictts", "kathbath", "kathbath_noisy", "commonvoice", "fleurs", "mucs"]
    all_records = []
    for b in benchmarks:
        sc = output_dir / f"{b}_summary.csv"
        if sc.exists():
            with open(sc, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    all_records.append(r)

    if not all_records:
        return

    master_csv = output_dir / "all_vistaar_comparison.csv"
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    master_md = output_dir / "all_vistaar_report.md"
    with open(master_md, "w", encoding="utf-8") as f:
        f.write("# Master Vistaar Benchmark Comparison: Marathi ASR\n\n")
        f.write("Evaluation across all Vistaar Marathi benchmarks with **Greedy, Prefix Beam, and KenLM 5-gram Decoding Ablations**:\n\n")
        f.write("## Word Error Rate (WER %) Comparison Across Benchmarks\n\n")
        f.write("| Benchmark | Our (Greedy) | Our (Lexicon Beam) | Our (KenLM Def α=0.5) | Our (KenLM Soft α=0.15) | SraVaani 1.0 | Indic Conformer 600M | IndicWhisper (Paper) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        cer_rows = []
        for b in benchmarks:
            b_records = [r for r in all_records if r["benchmark"].lower() == b.lower()]
            if not b_records:
                continue
            g_r = next((r for r in b_records if "Greedy" in r["model_name"]), None)
            b_r = next((r for r in b_records if "Lexicon" in r["model_name"]), None)
            kd_r = next((r for r in b_records if "Default" in r["model_name"]), None)
            ks_r = next((r for r in b_records if "Soft" in r["model_name"]), None)
            s_r = next((r for r in b_records if "SraVaani" in r["model_name"]), None)
            i_r = next((r for r in b_records if "Indic Conformer" in r["model_name"]), None)

            g_wer = f"{float(g_r['wer_percent']):.2f}%" if g_r else "N/A"
            b_wer = f"{float(b_r['wer_percent']):.2f}%" if b_r else "N/A"
            kd_wer = f"{float(kd_r['wer_percent']):.2f}%" if kd_r else "N/A"
            ks_wer = f"{float(ks_r['wer_percent']):.2f}%" if ks_r else "N/A"
            s_wer = f"{float(s_r['wer_percent']):.2f}%" if s_r else "N/A"
            i_wer = f"{float(i_r['wer_percent']):.2f}%" if i_r else "N/A"
            iw_wer = f"{INDICWHISPER_WER.get(b, 'N/A')}%"

            g_cer = f"{float(g_r['cer_percent']):.2f}%" if g_r else "N/A"
            b_cer = f"{float(b_r['cer_percent']):.2f}%" if b_r else "N/A"
            kd_cer = f"{float(kd_r['cer_percent']):.2f}%" if kd_r else "N/A"
            ks_cer = f"{float(ks_r['cer_percent']):.2f}%" if ks_r else "N/A"
            s_cer = f"{float(s_r['cer_percent']):.2f}%" if s_r else "N/A"
            i_cer = f"{float(i_r['cer_percent']):.2f}%" if i_r else "N/A"

            b_name = "Kathbath Hard" if b == "kathbath_noisy" else b.capitalize()
            f.write(f"| **{b_name}** | {g_wer} | {b_wer} | {kd_wer} | **{ks_wer}** | {s_wer} | {i_wer} | {iw_wer} |\n")
            cer_rows.append(f"| **{b_name}** | {g_cer} | {b_cer} | {kd_cer} | **{ks_cer}** | {s_cer} | {i_cer} |")

        f.write("\n\n## Character Error Rate (CER %) Comparison\n\n")
        f.write("| Benchmark | Our (Greedy) | Our (Lexicon Beam) | Our (KenLM Def α=0.5) | Our (KenLM Soft α=0.15) | SraVaani 1.0 | Indic Conformer 600M |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cr in cer_rows:
            f.write(cr + "\n")

    print(f"\n[Master Report] Saved master comparison: {master_md}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["indictts", "kathbath", "kathbath_noisy", "commonvoice", "fleurs", "mucs"])
    parser.add_argument("--output_dir", type=str, default="inference/results/vistaar")
    parser.add_argument("--cleanup", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in args.benchmarks:
        print(f"\n{'='*70}")
        print(f"PROCESSING VISTAAR BENCHMARK: {bench}")
        print(f"{'='*70}")

        url = VISTAAR_URLS[bench]
        bench_dir = extract_marathi_benchmark(bench, url, base_out_dir="data/vistaar_benchmarks", max_workers=24)

        manifest_path = bench_dir / "marathi" / "manifest.json"
        if not manifest_path.exists():
            manifest_path = Path("data/vistaar_benchmarks") / bench / "marathi" / "manifest.json"

        if not manifest_path.exists():
            print(f"[Error] Manifest not found at {manifest_path}!")
            continue

        evaluate_benchmark(bench, manifest_path, out_dir)
        print(f"[Storage] Preserving extracted benchmark on disk: {bench_dir}")
        build_master_vistaar_report(out_dir)


if __name__ == "__main__":
    main()