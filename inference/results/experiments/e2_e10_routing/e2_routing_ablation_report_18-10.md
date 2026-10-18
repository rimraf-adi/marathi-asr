# [2026-10-18] Experiment E2 Report: Routing Ablation Suite
**Date:** 2026-10-18 (18 October 2026)  
**Test Set:** IISc RESPIN meta_test_mr.json (2,170 utterances)  
**Checkpoint:** 34.14M MoE Conformer (Stage 3 Best Model)  
## 1. Routing Ablation Results (WER % per Dialect)

| Routing Condition | D1 (Malvani) | D2 (Ahirani) | D3 (Standard) | D4 (Varhadi) | Aggregate (ALL) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Learned Router (Default)** | 37.91% | 33.78% | 21.74% | 33.25% | **31.64%** |
| **Oracle Routing (Ground Truth)** | 43.98% | 40.93% | 31.26% | 40.47% | **39.13%** |
| **Random Expert Routing** | 46.69% | 43.25% | 27.36% | 43.03% | **40.02%** |
| **Shared Trunk Only (Zero MoE)** | 54.16% | 49.54% | 31.26% | 50.25% | **46.22%** |

## 2. Gating Router Dialect Classification Accuracy

- **Non-Standard Dialect Router Accuracy:** **43.47%** (702/1615)

| True Dialect | Predicted D1 Expert | Predicted D2 Expert | Predicted D4 Expert |
| :--- | :--- | :--- | :--- |
| **D1** | 97 | 441 | 21 |
| **D2** | 13 | 503 | 24 |
| **D4** | 21 | 393 | 102 |

## 3. Hypothesis E2a & E2b Verdicts

- **Learned WER:** 31.64%  
- **Oracle WER:** 39.13%  
- **Random Routing WER:** 40.02%  
- **Shared Trunk Only WER:** 46.22%  

> [!IMPORTANT]
> **Hypothesis E2a Verdict:** **REFUTED**
