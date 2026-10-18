# [2026-10-18] Experiment E11 Report: Error-Type Analysis & Dialect Standardization
**Date:** 2026-10-18 (18 October 2026)  
**Test Set:** IISc RESPIN meta_test_mr.json (2,170 utterances)  
## 1. Edit Operation Breakdown (Substitutions, Deletions, Insertions)

| Dialect | Model | Ref Words | Hits | Subs (%) | Dels (%) | Ins (%) | Total WER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | Our MoE | 4,556 | 2,855 | 32.99% (1503) | 4.35% (198) | 2.11% (96) | **39.44%** |
| **D1** | IndicConformer | 4,556 | 3,052 | 30.95% (1410) | 2.06% (94) | 1.58% (72) | **34.59%** |
| **D1** | SraVaani | 4,556 | 3,615 | 19.21% (875) | 1.45% (66) | 1.05% (48) | **21.71%** |
| **D2** | Our MoE | 4,573 | 3,105 | 27.60% (1262) | 4.50% (206) | 1.86% (85) | **33.96%** |
| **D2** | IndicConformer | 4,573 | 3,502 | 18.06% (826) | 5.36% (245) | 1.22% (56) | **24.64%** |
| **D2** | SraVaani | 4,573 | 3,806 | 12.62% (577) | 4.15% (190) | 0.98% (45) | **17.76%** |
| **D3** | Our MoE | 4,304 | 3,371 | 19.40% (835) | 2.28% (98) | 1.21% (52) | **22.89%** |
| **D3** | IndicConformer | 4,304 | 3,817 | 8.11% (349) | 3.21% (138) | 0.26% (11) | **11.57%** |
| **D3** | SraVaani | 4,304 | 3,924 | 6.16% (265) | 2.67% (115) | 0.14% (6) | **8.97%** |
| **D4** | Our MoE | 4,405 | 2,994 | 28.15% (1240) | 3.88% (171) | 2.04% (90) | **34.07%** |
| **D4** | IndicConformer | 4,405 | 3,369 | 19.25% (848) | 4.27% (188) | 0.77% (34) | **24.29%** |
| **D4** | SraVaani | 4,405 | 3,820 | 10.60% (467) | 2.68% (118) | 0.86% (38) | **14.14%** |
| **ALL** | Our MoE | 17,838 | 12,325 | 27.13% (4840) | 3.77% (673) | 1.81% (323) | **32.72%** |
| **ALL** | IndicConformer | 17,838 | 13,740 | 19.25% (3433) | 3.73% (665) | 0.97% (173) | **23.94%** |
| **ALL** | SraVaani | 17,838 | 15,165 | 12.24% (2184) | 2.74% (489) | 0.77% (137) | **15.75%** |

## 2. Dialect Standardization Substitution Analysis

Standardization substitutions occur when a dialect-specific form (e.g. *हाय*, *नाय*, *व्हय*, *माका*) is recognized as the canonical Standard Marathi form (*आहे*, *नाही*, *होय*, *मला*).

| Dialect | Total Substitutions (MoE) | Total Substitutions (Indic) | Extra Subs (Indic - MoE) | Standardization Subs (MoE) | Standardization Subs (Indic) | Extra Standardization Subs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | 1,503 | 1,410 | **-93** | 2 | 12 | **+10** |
| **D2** | 1,262 | 826 | **-436** | 0 | 0 | **+0** |
| **D3** | 835 | 349 | **-486** | 1 | 1 | **+0** |
| **D4** | 1,240 | 848 | **-392** | 1 | 21 | **+20** |
| **ALL** | 4,840 | 3,433 | **-1,407** | 4 | 34 | **+30** |

### Frequent Dialect Standardization Rewrites by IndicConformer

| Dialect | Dialect Form (Ref) | Standard Marathi Rewritten (Hyp) | Frequency in IndicConformer | Frequency in Our MoE |
| :--- | :--- | :--- | :--- | :--- |
| D1 | **नाय** | **नाही** | 5 | 0 |
| D1 | **माका** | **मला** | 4 | 0 |
| D1 | **येक** | **एक** | 3 | 2 |
| D4 | **हाय** | **आहे** | 10 | 0 |
| D4 | **हाये** | **आहे** | 9 | 1 |
| D4 | **व्हते** | **होते** | 2 | 0 |

## 3. Hypothesis E11 Verdict

- **Total Standardization Substitutions in IndicConformer:** 34
- **Total Standardization Substitutions in Our MoE:** 4
- **Reduction in Standardization Errors by MoE:** 30 errors prevented.

> [!NOTE]
> **Hypothesis E11 Verdict:** **REFUTED** (Standardization errors are systematically higher in the baseline multilingual model due to normative standard Marathi pretraining bias, whereas MoE experts preserve dialect phonemic forms).
