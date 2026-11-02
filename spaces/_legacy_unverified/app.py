import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# ZeroGPU Initialization
# -----------------------------------------------------------------------------
try:
    import spaces
    if hasattr(spaces, "GPU"):
        gpu = spaces.GPU
    else:
        def gpu(*a, **k):
            return (lambda f: f) if not (a and callable(a[0])) else a[0]
except Exception:
    def gpu(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@gpu(duration=10)
def gpu_status() -> str:
    """GPU-attached call so Hugging Face ZeroGPU runtime validates container startup."""
    try:
        import torch
        return f"ZeroGPU Online | PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}"
    except ImportError:
        return "ZeroGPU Online | PyTorch not installed"


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
RESULTS_REPO = os.getenv("RESULTS_REPO", "rimrafadii/marathi-asr-results")
LOCAL_RESULTS = os.getenv("LOCAL_RESULTS")

DATASETS = {
    "respin_mr": "RESPIN (Dialects)",
    "kathbath_mr": "Kathbath",
    "common_voice_mr": "Common Voice",
    "fleurs_mr": "FLEURS",
    "vaani_mr": "Vaani",
}
ALL_DATASET_KEYS = list(DATASETS.keys())


def find_local_results_file() -> Optional[str]:
    """Search for a local results.jsonl file."""
    candidates = [
        LOCAL_RESULTS,
        "results.jsonl",
        os.path.join(os.path.dirname(__file__), "results.jsonl"),
        os.path.join(os.path.dirname(__file__), "..", "spaces", "results.jsonl"),
        os.path.join(os.path.dirname(__file__), "..", "results.jsonl"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def load_raw_records() -> List[dict]:
    """Load evaluation records from HF Hub or local file fallback."""
    local_path = find_local_results_file()
    token = os.getenv("HF_TOKEN")

    if local_path and os.path.exists(local_path):
        records = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            return records

    try:
        path = hf_hub_download(
            repo_id=RESULTS_REPO,
            filename="results.jsonl",
            repo_type="dataset",
            token=token,
        )
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    except Exception as e:
        print(f"Warning: Could not fetch remote results: {e}. Falling back to empty or local.")
        if local_path and os.path.exists(local_path):
            records = []
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records
        return []


def parse_dataframe(records: List[dict]) -> pd.DataFrame:
    """Transform JSON records into a tabular pandas DataFrame."""
    rows = []
    for r in records:
        scores = r.get("scores", {})
        row = {
            "model": r.get("model", "Unknown"),
            "url": r.get("url", ""),
            "type": r.get("type", "open"),
            "params_b": r.get("params_b"),
            "rtfx": r.get("rtfx"),
            "license": r.get("license", "Unknown"),
            "hardware": r.get("hardware", "Fixed Standard"),
            "trained_on": r.get("trained_on") or [],
            "raw_record": r,
        }
        for ds_id in DATASETS:
            score_data = scores.get(ds_id)
            if score_data and "wer" in score_data:
                row[ds_id] = score_data["wer"]
                row[f"{ds_id}_ci95"] = score_data.get("wer_ci95", [None, None])
            else:
                row[ds_id] = float("nan")
                row[f"{ds_id}_ci95"] = [None, None]
        rows.append(row)

    df = pd.DataFrame(rows)
    for ds_id in DATASETS:
        if ds_id not in df:
            df[ds_id] = float("nan")
    return df


# -----------------------------------------------------------------------------
# View Generation Helpers
# -----------------------------------------------------------------------------
def get_main_leaderboard(search_query: str = "") -> pd.DataFrame:
    records = load_raw_records()
    if not records:
        return pd.DataFrame()

    df = parse_dataframe(records)

    # Compute macro average across all available benchmark datasets
    available_cols = [c for c in ALL_DATASET_KEYS if c in df.columns]
    df["avg_metric"] = df[available_cols].mean(axis=1, skipna=True)
    df = df.sort_values("avg_metric", ascending=True).reset_index(drop=True)

    # Add medal rank icons
    def format_rank(idx):
        if idx == 0:
            return "🥇 1"
        elif idx == 1:
            return "🥈 2"
        elif idx == 2:
            return "🥉 3"
        return f"   {idx + 1}"

    df["Rank"] = [format_rank(i) for i in range(len(df))]

    if search_query:
        terms = [t.strip().lower() for t in search_query.split(",") if t.strip()]
        df = df[df["model"].str.lower().apply(lambda m: any(t in m for t in terms))]

    df["Model"] = df["model"]
    df["Macro Avg WER (%)"] = df["avg_metric"].round(2)
    df["RESPIN (Dialects)"] = df["respin_mr"].apply(lambda v: f"{v:.2f}%" if pd.notnull(v) else "-")
    df["Kathbath"] = df["kathbath_mr"].apply(lambda v: f"{v:.2f}%" if pd.notnull(v) else "-")
    df["Common Voice"] = df["common_voice_mr"].apply(lambda v: f"{v:.2f}%" if pd.notnull(v) else "-")
    df["FLEURS"] = df["fleurs_mr"].apply(lambda v: f"{v:.2f}%" if pd.notnull(v) else "-")
    df["Vaani"] = df["vaani_mr"].apply(lambda v: f"{v:.2f}%" if pd.notnull(v) else "-")
    df["Speed (RTFx)"] = df["rtfx"].apply(lambda s: f"⚡ {s:.1f}x" if pd.notnull(s) else "-")
    df["Params (B)"] = df["params_b"].apply(lambda p: f"{p:.3f}B" if pd.notnull(p) else "-")
    df["License"] = df["license"]

    display_cols = [
        "Rank",
        "Model",
        "Macro Avg WER (%)",
        "RESPIN (Dialects)",
        "Kathbath",
        "Common Voice",
        "FLEURS",
        "Vaani",
        "Speed (RTFx)",
        "Params (B)",
        "License",
    ]
    return df[display_cols]


def get_dialect_track_table() -> pd.DataFrame:
    """Exact RESPIN Dialect Benchmark breakdown table."""
    records = load_raw_records()
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        slices = r.get("slices", {}).get("respin_mr", {}).get("dialect", {})
        if not slices:
            continue

        respin_score = r.get("scores", {}).get("respin_mr", {}).get("wer")
        params_b = r.get("params_b", 0)

        # Active params estimation
        if "marathi-conformer-130m-moe" in r.get("model", ""):
            active_p = "98.9M (Top-1)"
            total_p = "130.8M"
        elif "marathi-conformer-33m-moe" in r.get("model", ""):
            active_p = "25.2M (Top-1)"
            total_p = "33.8M"
        elif "marathi-conformer-130m-dense" in r.get("model", ""):
            active_p = "130.8M"
            total_p = "130.8M"
        elif "marathi-conformer-33m-dense" in r.get("model", ""):
            active_p = "20.5M"
            total_p = "20.5M"
        elif "indic-conformer-600m" in r.get("model", ""):
            active_p = "600M"
            total_p = "600M"
        elif "SraVaani" in r.get("model", ""):
            active_p = "100M"
            total_p = "100M"
        else:
            active_p = f"{params_b*1000:.0f}M" if params_b else "-"
            total_p = f"{params_b*1000:.0f}M" if params_b else "-"

        row = {
            "Model": r.get("model", "Unknown"),
            "Total Params": total_p,
            "Active Params": active_p,
            "D1 (Malvani)": f"{slices.get('Malvani (D1)', {}).get('wer', float('nan')):.2f}%",
            "D2 (Ahirani)": f"{slices.get('Ahirani (D2)', {}).get('wer', float('nan')):.2f}%",
            "D3 (Standard)": f"{slices.get('Standard (D3)', {}).get('wer', float('nan')):.2f}%",
            "D4 (Varhadi)": f"{slices.get('Varhadi (D4)', {}).get('wer', float('nan')):.2f}%",
            "Aggregate RESPIN WER": respin_score if respin_score is not None else float("nan"),
            "RTFx": r.get("rtfx"),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Aggregate RESPIN WER", ascending=True).reset_index(drop=True)
    df["Rank"] = [f"🥇 1" if i == 0 else f"🥈 2" if i == 1 else f"🥉 3" if i == 2 else f"   {i+1}" for i in range(len(df))]
    df["Aggregate WER (%)"] = df["Aggregate RESPIN WER"].apply(lambda v: f"{v:.2f}%")
    df["Inference Speed"] = df["RTFx"].apply(lambda s: f"⚡ {s:.1f}x" if pd.notnull(s) else "-")

    cols = [
        "Rank",
        "Model",
        "Total Params",
        "Active Params",
        "D1 (Malvani)",
        "D2 (Ahirani)",
        "D3 (Standard)",
        "D4 (Varhadi)",
        "Aggregate WER (%)",
        "Inference Speed",
    ]
    return df[cols]


def get_dialect_chart() -> go.Figure:
    records = load_raw_records()
    chart_data = []
    dialects = ["Malvani (D1)", "Ahirani (D2)", "Standard (D3)", "Varhadi (D4)"]
    for r in records:
        slices = r.get("slices", {}).get("respin_mr", {}).get("dialect", {})
        if not slices:
            continue
        model_name = r.get("model", "Unknown").split("/")[-1]
        for d in dialects:
            wer = slices.get(d, {}).get("wer")
            if wer is not None:
                chart_data.append({
                    "Model": model_name,
                    "Dialect": d,
                    "WER (%)": wer,
                })

    if not chart_data:
        return go.Figure()

    df_plot = pd.DataFrame(chart_data)
    fig = px.bar(
        df_plot,
        x="Dialect",
        y="WER (%)",
        color="Model",
        barmode="group",
        title="Word Error Rate Across Regional Marathi Dialects (Lower is Better)",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def get_pareto_plot() -> go.Figure:
    records = load_raw_records()
    if not records:
        return go.Figure()

    rows = []
    for r in records:
        scores = r.get("scores", {})
        respin_wer = scores.get("respin_mr", {}).get("wer")
        rtfx = r.get("rtfx")
        if respin_wer and rtfx:
            rows.append({
                "model": r.get("model", "Unknown").split("/")[-1],
                "full_name": r.get("model", "Unknown"),
                "rtfx": rtfx,
                "respin_wer": respin_wer,
                "params_b": r.get("params_b", 0.1),
            })

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows)
    df["bubble_size"] = df["params_b"].fillna(0.1).apply(lambda p: max(16, min(50, p * 30 + 14)))

    fig = px.scatter(
        df,
        x="rtfx",
        y="respin_wer",
        hover_name="full_name",
        text="model",
        size="bubble_size",
        log_x=True,
        labels={
            "rtfx": "Inference Speed (RTFx) [Higher is Faster ⚡]",
            "respin_wer": "RESPIN Dialect WER (%) [Lower is Better 🎯]",
        },
        title="Dialect Accuracy vs. Real-Time Factor (Pareto Frontier)",
        color="respin_wer",
        color_continuous_scale="Viridis_r",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=12, family="Inter"))
    fig.update_layout(
        template="plotly_white",
        height=580,
        xaxis=dict(gridcolor="#f1f5f9", title_font=dict(size=13)),
        yaxis=dict(gridcolor="#f1f5f9", title_font=dict(size=13)),
    )
    return fig


def get_integrity_table() -> pd.DataFrame:
    records = load_raw_records()
    rows = []
    for r in records:
        trained = set(r.get("trained_on") or [])
        rows.append({
            "Model": r.get("model", "Unknown"),
            "Hardware Standard": r.get("hardware", "NVIDIA RTX A5000 / A6000"),
            "Trained on RESPIN?": "⚠️ Trained (In-Domain)" if "respin_mr" in trained else "✅ Unseen (Clean Zero-Shot)",
            "Trained on Kathbath?": "⚠️ Trained (In-Domain)" if "kathbath_mr" in trained else "✅ Unseen (Clean Zero-Shot)",
            "Trained on Vaani?": "⚠️ Trained (In-Domain)" if "vaani_mr" in trained else "✅ Unseen (Clean Zero-Shot)",
            "Trained on FLEURS?": "⚠️ Trained (In-Domain)" if "fleurs_mr" in trained else "✅ Unseen (Clean Zero-Shot)",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Aesthetic UI Layout
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
/* Aesthetic styling for Marathi ASR Leaderboard */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1350px !important;
    margin: auto !important;
}

.hero-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 32px 28px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    border: 1px solid #334155;
    text-align: center;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 15px;
    color: #94a3b8;
    max-width: 800px;
    margin: 0 auto 20px auto;
    line-height: 1.5;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 18px;
}

.stat-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    backdrop-filter: blur(8px);
}

.stat-val {
    font-size: 22px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
}

.gr-dataframe table {
    font-size: 13.5px !important;
}

.gr-dataframe th {
    background-color: #f8fafc !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    padding: 10px 14px !important;
}

.gr-dataframe td {
    padding: 10px 14px !important;
}
"""

with gr.Blocks(title="Marathi ASR Benchmark Leaderboard") as demo:
    # Inject styling
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    # Hero Banner & Highlight KPI Cards
    gr.HTML(
        """
        <div class="hero-banner">
            <div class="hero-title">🎙️ Marathi ASR Benchmark Leaderboard</div>
            <div class="hero-sub">
                Standardized, dialect-aware evaluation of Automatic Speech Recognition models across Marathi regional dialects and spoken benchmarks under the frozen <code>mr-norm-1</code> protocol.
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-val">19.04% WER</div>
                    <div class="stat-label">🏆 Best Dialect SOTA (RESPIN)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">⚡ 384.6x RTFx</div>
                    <div class="stat-label">🚀 Scaled MoE Throughput</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">98.9M / 130.8M</div>
                    <div class="stat-label">💡 Active / Total Parameters</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">mr-norm-1</div>
                    <div class="stat-label">🛡️ Frozen Normalization</div>
                </div>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        # TAB 1: ALL-IN-ONE LEADERBOARD
        with gr.Tab("🏆 Overall Leaderboard"):
            search_input = gr.Textbox(
                label="Search models",
                placeholder="Type model name (e.g. conformer, whisper, sravaani)...",
                show_label=False,
                max_lines=1,
            )
            main_table = gr.Dataframe(interactive=False, wrap=True)
            search_input.change(get_main_leaderboard, inputs=[search_input], outputs=[main_table])
            demo.load(get_main_leaderboard, inputs=[search_input], outputs=[main_table])

        # TAB 2: RESPIN REGIONAL DIALECT TRACK
        with gr.Tab("🔬 RESPIN Dialect Track (D1–D4)"):
            gr.Markdown(
                """
                ### 🎙️ Dialect-by-Dialect Breakdown (IISc RESPIN Held-Out Suite)
                Evaluated strictly across **160 unseen test speakers** (2,170 utterances / 3.04 hours):
                - **D1 (Malvani / Konkan)**: Coastal phonology, retroflex variations.
                - **D2 (Ahirani / Khandesh)**: Northern border dialect with Gujarati substrate.
                - **D3 (Standard Marathi)**: Broadcast and standard literary phonology.
                - **D4 (Varhadi / Vidarbha)**: Eastern Marathi with alveolar affricate realizations.
                """
            )
            dialect_table = gr.Dataframe(interactive=False, wrap=True)
            dialect_plot = gr.Plot()
            demo.load(get_dialect_track_table, None, dialect_table)
            demo.load(get_dialect_chart, None, dialect_plot)

        # TAB 3: SPEED VS ACCURACY PARETO
        with gr.Tab("⚡ Speed vs. Accuracy (Pareto)"):
            gr.Markdown(
                """
                ### 🚀 Inference Throughput (RTFx) vs. Dialect Accuracy (WER)
                Top-left points achieve both higher speedup and superior transcription accuracy.
                Bubble sizes scale with total model parameter footprint.
                """
            )
            pareto_chart = gr.Plot()
            demo.load(get_pareto_plot, None, pareto_chart)

        # TAB 4: CONTAMINATION & INTEGRITY
        with gr.Tab("🛡️ Contamination & Data Integrity"):
            gr.Markdown(
                """
                ### 🔍 Dataset Overlap & Training Corpus Transparency
                Tracking data leakage: Models trained on specific benchmark test prompts achieve lower error rates on those sets, while out-of-domain evaluation measures genuine zero-shot acoustic adaptability.
                """
            )
            integrity_table = gr.Dataframe(interactive=False, wrap=True)
            demo.load(get_integrity_table, None, integrity_table)

        # TAB 5: METHODOLOGY & ABOUT
        with gr.Tab("📖 Methodology & ZeroGPU"):
            gr.Markdown(
                """
                ### 📋 Benchmark Evaluation Protocol

                #### 1. Frozen Normalization: `mr-norm-1`
                - **Unicode Normalization Form C (NFC)** canonical decomposition and composition.
                - **Numeral Unification**: Maps Devanagari numerals (`०-९`) to ASCII (`0-9`) consistently across references and hypotheses.
                - **Stripping**: Removes zero-width joiners (`U+200B`, `U+200C`, `U+200D`), dandas (`।`, `॥`), and punctuation.
                - **Lowercasing**: Lowercases Latin letters to handle code-mixed English words fairly.

                #### 2. Macro-Average WER
                $$\\text{Macro Average WER} = \\frac{1}{K} \\sum_{k=1}^K \\text{WER}_k$$
                Weights each distinct audio domain equally to prevent high-utterance subsets from skewing the rank.

                #### 3. Hardware Standardization
                All inference timings and RTFx metrics are evaluated on **NVIDIA RTX A5000 / A6000 hardware** under identical FP16 PyTorch execution to ensure scientific comparability.
                """
            )
            with gr.Accordion("ZeroGPU Hardware Status", open=False):
                gpu_btn = gr.Button("Check ZeroGPU Status")
                gpu_text = gr.Textbox(label="Status", interactive=False)
                gpu_btn.click(gpu_status, outputs=gpu_text)

if __name__ == "__main__":
    demo.launch()
