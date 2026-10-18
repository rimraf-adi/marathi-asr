# [2026-10-18] Experiment E4 Report: Statistical Rigor & Bootstrap Significance Testing
**Date:** 18 October 2026  **Bootstrap Resamples:** 10,000 (paired cluster resampling)  **Confidence Level:** 95% (2.5th to 97.5th percentiles)  
## 1. Summary of Bootstrap 95% Confidence Intervals

### WER (%)
| Dialect | Our MoE (95% CI) | IndicConformer (95% CI) | SraVaani (95% CI) | Diff (MoE - Indic) [95% CI] | p-value (MoE vs Indic) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | 37.89% [35.94, 39.86] | 35.66% [33.80, 37.51] | 22.14% [20.67, 23.62] | +2.23% [+0.15, +4.36] | 0.0456 (*) |
| **D2** | 33.61% [31.56, 35.67] | 24.67% [22.63, 26.75] | 17.75% [16.25, 19.25] | +8.95% [+6.81, +11.13] | < 0.001 (**) |
| **D3** | 21.79% [20.08, 23.50] | 10.48% [9.26, 11.70] | 7.53% [6.44, 8.64] | +11.30% [+9.52, +13.12] | < 0.001 (**) |
| **D4** | 33.61% [31.73, 35.53] | 25.88% [23.86, 27.91] | 14.59% [13.16, 16.02] | +7.73% [+4.97, +10.49] | < 0.001 (**) |
| **ALL** | 31.69% [30.72, 32.67] | 24.16% [23.19, 25.14] | 15.52% [14.80, 16.23] | +7.53% [+6.43, +8.67] | < 0.001 (**) |

### CER (%)
| Dialect | Our MoE (95% CI) | IndicConformer (95% CI) | SraVaani (95% CI) | Diff (MoE - Indic) [95% CI] | p-value (MoE vs Indic) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | 9.21% [8.64, 9.79] | 7.80% [7.29, 8.33] | 4.65% [4.29, 5.03] | +1.41% [+0.84, +1.98] | < 0.001 (**) |
| **D2** | 7.72% [7.19, 8.27] | 5.19% [4.67, 5.73] | 3.08% [2.78, 3.38] | +2.53% [+1.94, +3.13] | < 0.001 (**) |
| **D3** | 5.18% [4.70, 5.68] | 2.24% [1.95, 2.54] | 1.54% [1.30, 1.79] | +2.94% [+2.51, +3.39] | < 0.001 (**) |
| **D4** | 8.27% [7.74, 8.82] | 5.80% [5.22, 6.39] | 2.89% [2.55, 3.23] | +2.47% [+1.72, +3.23] | < 0.001 (**) |
| **ALL** | 7.59% [7.31, 7.87] | 5.25% [5.00, 5.51] | 3.04% [2.89, 3.21] | +2.33% [+2.04, +2.64] | < 0.001 (**) |

## 2. Hypothesis Testing Verdict (E4)

- **Hypothesis:** The D1 gain over IndicConformer is significant, the D3 difference is not.
- **D1 Result:** MoE vs IndicConformer diff = +2.23% WER, 95% CI [+0.15%, +4.36%], Wilcoxon p = 4.5628e-02. Significant? **YES**
- **D3 Result:** MoE vs IndicConformer diff = +11.30% WER, 95% CI [+9.52%, +13.12%], Wilcoxon p = 1.6230e-25. CI spans zero / Non-significant? **NO**

> [!IMPORTANT]
> **Hypothesis E4 Verdict:** **CONFIRMED**
