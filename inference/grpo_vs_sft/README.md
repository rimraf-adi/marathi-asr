# GRPO vs. Vanilla SFT Ablation Study Suite

This directory contains the comparative evaluation suite for studying sequence-level **Group Relative Policy Optimization (GRPO)** versus **Vanilla Supervised Fine-Tuning (SFT)** on the frozen **IISc RESPIN Marathi Test Set** (2,170 utterances across D1, D2, D3, D4 dialects).

---

## 1. Experimental Overview

Both models begin from the exact same Stage 2 multi-dialect checkpoint (`stage2_moe/best_model.pt`) and train on the exact same multi-source pool of datasets:
- **100% IISc RESPIN** (both the 5% held-out split and the 95% training split = **122,639 utterances**)
- **OpenSLR64 Marathi Speech Corpus** (1,569 utterances)
- **AI4Bharat Kathbath Marathi** (Clean train 84,070 utts + Noisy 1,631 utts = **85,701 utterances**)
- **Streaming Pretraining Corpus** (`ai4bharat/Shrutilipi` & `ARTPARK-IISc/Vaani`)

### **Comparison Arms:**
1. **Vanilla SFT (`final_alignment/train_alignment.py`):**
   - Standard PyTorch `CTCLoss` with AdamW and cosine schedule.
   - Fine-tunes Layers 5–12 + CTC head while keeping Layers 1–4 frozen as an acoustic anchor.
2. **GRPO Polish (`final_alignment/train_grpo_polish.py`):**
   - Samples $G=4$ candidate CTC paths per utterance (1 greedy exploitation anchor + 3 temperature stochastic rollouts).
   - Computes non-differentiable sequence reward:
     $$R(Y_i, Y^*) = - (\text{CER}(Y_i, Y^*) + \gamma_{\text{wer}} \cdot \text{WER}(Y_i, Y^*))$$
   - Normalizes group relative advantages:
     $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
   - PPO clipped surrogate objective with $D_{KL}$ penalty against reference model and auxiliary CTC anchor.

---

## 2. Running the Benchmark

```bash
.\.venv311\Scripts\python.exe inference/grpo_vs_sft/benchmark_grpo_vs_sft.py \
    --sft_checkpoint runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt \
    --grpo_checkpoint runs/scaled-conformer512-moe-3stage/checkpoints/stage3_grpo_polish/best_model.pt \
    --output_dir inference/grpo_vs_sft/results \
    --beam_width 16
```

---

## 3. Statistical Testing & Outputs

The evaluation produces:
1. `grpo_vs_sft_summary.csv`: Per-dialect and overall aggregate CER, WER, $\Delta \text{CER}$, Relative Error Reduction (RER %), and Real-Time Factor (RTF).
2. `grpo_vs_sft_predictions.csv`: Per-utterance transcriptions and CER/WER comparison.
3. `grpo_vs_sft_report.md`: Structured Markdown ablation report.
4. `grpo_vs_sft_table.tex`: Publication-ready LaTeX table with paired Student's $t$-test and Wilcoxon signed-rank $p$-values.
