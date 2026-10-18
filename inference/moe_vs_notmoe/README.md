# Empirical Benchmark: Dialect-Aware MoE vs. Dense Baseline (MoE vs. Not-MoE)

## 1. Central Research Question
> **Is dialect-aware Mixture-of-Experts necessary for Marathi ASR, or is a pooled dense model sufficient?**

This is an empirical question that cannot be answered by theoretical assertion. It is settled through a rigorous **baseline ladder** evaluated on the frozen **IISc RESPIN Held-Out Test Set** (2,170 utterances across 4 regional dialects):

1. **Dense Baseline (MoE-Free)**: All Marathi dialects pooled, standard 12-layer Conformer ($d_{\text{model}}=512$, 63.8M parameters).
2. **Dialect Conditioning / Adapters**: Per-dialect conditioning / LoRA adapters.
3. **Dialect-Balanced Resampling**: Balanced data distribution addressing resource skew without routing.
4. **Dialect-Aware Adaptive-Sparsity MoE**: Conditional computation with top-1 dynamic routing across regional experts (Malvani, Ahirani, Varhadi).

## 2. The Negative Transfer Test
The one result that would make MoE **necessary**, not just sufficient:
- **Evidence of Negative Transfer**: A single dense model trained on pooled Marathi dialects exhibits cross-dialect gradient interference, underperforming on colloquial regional dialects (D1 Malvani, D2 Ahirani, D4 Varhadi) relative to Standard Marathi (D3).
- **Disjoint Capacity Advantage**: MoE routes conflicting acoustic/phonetic distributions to specialized expert pathways at matched active parameter cost ($63.8\text{M}$ active parameters per token).
- **The Honest Scientific Verdict**:
  - If MoE significantly outperforms Dense ($p < 0.05$ on Wilcoxon/t-test) and closes the negative transfer gap: MoE necessity is defended with empirical evidence.
  - If Dense matches or beats MoE at matched active parameters: we drop MoE, adopt the simpler dense architecture, and report the findings cleanly.

## 3. Test Set Breakdown (IISc RESPIN Frozen Test Set)
- **Total Utterances**: 2,170
- **D1 (Malvani / Konkan)**: 559 samples
- **D2 (Ahirani / Khandesh)**: 540 samples
- **D3 (Standard Marathi)**: 555 samples
- **D4 (Varhadi / Vidarbha)**: 516 samples

## 4. Running the Benchmark

### Automatic Execution (Both Models):
```powershell
python inference/moe_vs_notmoe/benchmark_moe_vs_dense.py `
  --moe_checkpoint runs/scaled-conformer512-moe-3stage/checkpoints/stage3_final_alignment/best_model.pt `
  --dense_checkpoint runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt `
  --output_dir inference/moe_vs_notmoe/results
```

### Outputs Generated:
- `results/respin_moe_vs_dense_summary.csv`: Summary metrics across architectures and dialects.
- `results/respin_moe_vs_dense_detailed.csv`: Utterance-level predictions, references, CER, WER, and per-sample delta.
- `results/moe_vs_dense_scientific_report.md`: Markdown report detailing dialect breakdown, negative transfer metrics, and statistical hypothesis tests.
- `results/table_respin_benchmark.tex`: Publication-ready LaTeX table for paper inclusion.
