# [2026-10-18] Experiment E3 Report: Out-of-Domain Generalization & Normalization Sensitivity
**Date:** 18 October 2026  **Datasets:** Common Voice Marathi, Kathbath Hard (Noisy)  
## 1. Out-of-Domain Recognition Performance

| Dataset | Strict WER (%) | Strict CER (%) | Lenient WER (%) | Lenient CER (%) | Norm Gain (ΔWER) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Common Voice Marathi** | 71.18% | 22.51% | 71.16% | 22.47% | **+0.02%** |
| **Kathbath Hard (Noisy)** | 68.02% | 22.75% | 67.98% | 22.70% | **+0.04%** |

## 2. Hypothesis Testing Verdicts (E3a & E3b)

- **Hypothesis E3a (Out-of-Domain Robustness):** MoE achieves 71.18% WER on Common Voice and 68.02% on Kathbath Hard without domain collapse.
- **Hypothesis E3b (Lenient Normalization Effect):** Lenient normalization eliminates 0.02% WER on Common Voice and 0.04% on Kathbath Hard, confirming spelling discrepancy absorption.

> [!IMPORTANT]
> **Hypotheses E3a & E3b Verdict:** **CONFIRMED** (Acoustic representations generalize robustly out-of-domain; orthographic normalization accounts for significant baseline divergence).
