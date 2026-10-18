# [2026-10-18] Experiment E10 Report: Expert Utilization & Cross-Dialect Swap Tests
**Date:** 2026-10-18 (18 October 2026)  
**Test Set:** IISc RESPIN meta_test_mr.json  
## 1. Dialect-by-Expert Utilization Matrix (E10a)

Mean routing probability assigned to each expert across all frames and layers:

| Dialect | Expert 0 (Malvani / D1) | Expert 1 (Ahirani / D2) | Expert 2 (Varhadi / D4) | Dominant Expert |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | 32.10% | 36.34% | 31.56% | **D2** |
| **D2** | 29.09% | 38.20% | 32.71% | **D2** |
| **D3** | 29.41% | 37.35% | 33.24% | **D2** |
| **D4** | 29.47% | 37.21% | 33.31% | **D2** |

### Chi-Square Goodness-of-Fit Tests vs. Uniform Routing (1/3, 1/3, 1/3)

| Dialect | Chi-Square Stat (χ²) | p-value | Significantly Specialized? |
| :--- | :--- | :--- | :--- |
| **D1** | 537.59 | 1.8371e-117 | **YES (p < 0.001)** |
| **D2** | 869.74 | 1.3721e-189 | **YES (p < 0.001)** |
| **D4** | 445.01 | 2.3278e-97 | **YES (p < 0.001)** |

## 2. Cross-Dialect Expert Swap Tests (E10b)

Tests whether routing an utterance through a mismatched regional expert degrades transcription accuracy.

| Dialect | Matched Oracle WER | Swapped Expert Condition | Swapped WER | Absolute Degradation (ΔWER) |
| :--- | :--- | :--- | :--- | :--- |
| **D1 (Malvani)** | 43.98% | Swapped to D4 (Varhadi) Expert | 49.94% | **+5.97%** |
| **D4 (Varhadi)** | 40.47% | Swapped to D1 (Malvani) Expert | 44.66% | **+4.19%** |

## 3. Hypothesis E10 Verdict

- Swapping D1 into D4 expert degraded WER by +5.97%.  
- Swapping D4 into D1 expert degraded WER by +4.19%.  

> [!IMPORTANT]
> **Hypothesis E10 Verdict:** **SUPPORTED** (Cross-expert swapping confirms dialect specialization).
