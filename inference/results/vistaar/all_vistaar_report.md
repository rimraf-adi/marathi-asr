# Master Vistaar Benchmark Comparison: Marathi ASR

Evaluation across all Vistaar Marathi benchmarks with **Greedy, Prefix Beam, and KenLM 5-gram Decoding Ablations**:

## Word Error Rate (WER %) Comparison Across Benchmarks

| Benchmark | Our (Greedy) | Our (Lexicon Beam) | Our (KenLM Def α=0.5) | Our (KenLM Soft α=0.15) | SraVaani 1.0 | Indic Conformer 600M | IndicWhisper (Paper) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Indictts** | 53.00% | 51.60% | 48.50% | **50.90%** | 14.10% | 13.40% | 11.4% |
| **Kathbath** | N/A | N/A | N/A | **N/A** | 17.26% | 17.61% | 19.9% |
| **Kathbath Hard** | 69.95% | 69.68% | 61.71% | **69.56%** | 18.98% | 18.86% | 22.1% |


## Character Error Rate (CER %) Comparison

| Benchmark | Our (Greedy) | Our (Lexicon Beam) | Our (KenLM Def α=0.5) | Our (KenLM Soft α=0.15) | SraVaani 1.0 | Indic Conformer 600M |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Indictts** | 12.62% | 12.25% | 13.08% | **12.09%** | 3.36% | 2.84% |
| **Kathbath** | N/A | N/A | N/A | **N/A** | 5.59% | 5.63% |
| **Kathbath Hard** | 25.62% | 25.29% | 25.84% | **24.87%** | 6.29% | 6.15% |
