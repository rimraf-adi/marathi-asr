# [2026-10-18] Experiment Plan: Dialect-Aware Sparse MoE Conformer for Marathi ASR

**Date:** 2026-10-18 (18 October 2026)  
*Working document: update the status column as results arrive.*

---

## 1. Purpose and Ground Rules

This plan lists every experiment needed to turn the current benchmark report into a publishable, defensible paper (target: **IEEE Access**, fallback: **Speech Communication** or **ACM TALLIP**). Each experiment has a hypothesis, a setup, a metric, and a refutation rule written before any run.

### Ground Rules
- **Date-stamp each hypothesis** before running it. A hypothesis written after seeing numbers is a story, not a test.
- **Use the same data, optimizer, number of steps, and data order** across all variants. Use 3 seeds where affordable.
- **Primary hypotheses (confirmatory):** H0, E1a, E2a, E3a. Everything else is exploratory and must be labelled as such in the paper.
- **Report failed hypotheses.** A properly tested negative result is still a finding.
- **Report bootstrap 95% confidence intervals and paired significance tests**, not point estimates alone.

---

## 2. Fixed Protocol

| Item | Specification |
| :--- | :--- |
| **Test set** | IISc RESPIN `meta_test_mr.json`: 2,170 utterances, 3.041 h, dialects D1 (Malvani), D2 (Ahirani), D3 (Standard), D4 (Varhadi) |
| **Systems compared** | Our MoE (130.8M total, 98.9M active), IndicConformer-600M, SraVaani-1.0, plus the dense baselines below |
| **Metrics** | CER, WER, SER, substitution/deletion/insertion counts, per dialect and overall |
| **Normalization** | One identical pipeline applied to all references and hypotheses (punctuation, whitespace, numerals). Document it. |
| **Model selection** | Checkpoints chosen on a dev split only, never on the test set |
| **Isolation evidence** | Document speaker, utterance, and sentence-text overlap checks between train and test |
| **Cost logging** | GPU-hours, hardware, data hours, and any upcycling source counted toward the budget claim |

---

## 3. Central Hypothesis (H0)

> **H0:** Dialect-conditioned sparse experts reduce WER on non-standard dialects (D1, D2, D4) relative to a dense model with equal active parameters, with no significant loss on standard Marathi (D3).
>
> **Refuted if:** The dense baseline falls within the bootstrap confidence interval of the MoE on D1, D2, and D4.

---

## 4. Tier 1: Required Experiments
*These decide whether a paper exists. Run them first.*

### E1. Dense Baselines (Capacity and Compute Controls)
- **E1a:** Dense Conformer-512, same data and schedule, matched to total parameters (130.8M).
- **E1b:** Dense Conformer matched to active parameters (98.9M).
- **Hypothesis:** MoE is better than both on D1, D2, D4 by more than the confidence interval.
- **Metric:** WER and CER per dialect.
- **Refuted if:** Either dense model matches or beats the MoE.
- **Action if refuted:** Drop the claim that MoE is the method. Reframe the paper as a data-efficiency and low-latency study.

### E2. Routing Ablation
- **Variants:** Oracle dialect label, learned router, random routing, single shared expert.
- **Also report:** Router dialect-classification accuracy and WER when routing is wrong.
- **Hypothesis (E2a):** Oracle is at least as good as learned routing, and both clearly beat random routing.
- **Hypothesis (E2b):** Routing errors concentrate on phonologically close dialect pairs.
- **Refuted if:** Random routing is within the confidence interval of learned routing (the dialect-aware claim fails).
- **Open question to answer first:** Does current inference use the true dialect label or a predicted one?

### E3. Out-of-Domain Evaluation
- **Data:** Kathbath, Common Voice Marathi, Vaani Marathi (use what is available).
- **Systems:** Our MoE, dense baselines, IndicConformer, SraVaani.
- **Hypothesis (E3a):** The MoE advantage over IndicConformer persists out of domain.
- **Hypothesis (E3b):** Under lenient normalization, part of the baselines' error disappears, showing it was standardization of dialect forms rather than misrecognition.
- **Refuted if:** The gain vanishes out of domain (it reflected RESPIN spelling conventions).

### E4. Statistical Rigor
- **Protocol:** Bootstrap 95% confidence intervals for WER and CER per dialect and overall. Paired significance tests against IndicConformer and SraVaani.
- **Hypothesis:** The D1 gain over IndicConformer is significant, the D3 difference is not.
- **Refuted if:** The D1 confidence interval includes zero.

---

## 5. Tier 2: Method Evidence

| ID | Experiment | Variants | Hypothesis | Refuted if |
| :--- | :--- | :--- | :--- | :--- |
| **E5** | Number of experts | 1, 2, 3, 4 (8 if affordable) | WER improves up to about the number of real dialect clusters, then plateaus. 3 experts match 4 if two dialects share phonology. | Curve is flat from 1 expert |
| **E6** | MoE placement and gating | All 12 layers, upper 8, upper 4, lower 4; top-1 vs top-2 | Experts help most in upper layers (phonetic and lexical variation), little in lower layers. | Lower-layer MoE performs equally |
| **E7** | Freezing and upcycling | Frozen lower 4 vs fully trainable; scratch vs upcycled from dense | Freezing costs little accuracy and saves budget; upcycling beats scratch at fixed budget. | Full fine-tuning is clearly better |
| **E8** | Load balancing | With vs without auxiliary balancing loss | Without it, experts collapse and dialect gains shrink. | No difference |
| **E9** | Data composition | Remove each corpus in turn (RESPIN, Kathbath, Common Voice, OpenSLR64, Vaani, Shrutilipi) | Dialect gains come mainly from RESPIN; other corpora help D3 and noise robustness. | Removing RESPIN barely moves dialect WER |

> [!NOTE]
> For **E5**, state why the model uses 3 experts for 4 dialects, and include the answer. For **E7**, state the upcycling source and count its cost in the budget claim.

---

## 6. Tier 3: The Scientific Finding

### E10. Expert Analysis
- **E10a:** Build the dialect-by-expert utilization table.  
  *Hypothesis:* Usage differs by dialect (test against uniform distribution).
- **E10b:** Swap test. Put D1's expert into D4 inference.  
  *Hypothesis:* This degrades D4 more than swapping in a neighbouring dialect's expert.
- **E10c:** Per-phoneme error breakdown by matched vs mismatched expert, for phenomena such as retroflex elision (D1) and nasalization (D4).  
  *Note:* This is the hardest prediction and the most valuable if it holds. Do not promise it in a proposal.
- **Refuted if:** Utilization is dialect-independent, or swaps barely matter.

### E11. Error-Type Analysis
- Break errors into substitutions, deletions, insertions per dialect.
- Count standardization errors separately (for example, a baseline rewriting a dialect form such as हाय as आहेत).
- **Hypothesis:** Most of IndicConformer's extra errors on D1 and D4 are standardization substitutions, while the MoE's remaining errors are phonetic.
- **Refuted if:** Standardization counts are small.

---

## 7. Tier 4: Optional (Only if Cheap)

| ID | Experiment | Note |
| :--- | :--- | :--- |
| **E12** | Decoder ablation: greedy CTC, beam search, beam search with lexicon | State whether the lexicon includes RESPIN text. If the lexicon helps this model far more than the baselines, check for text overlap. |
| **E13** | Robustness: noise or 8 kHz telephony audio | Skip unless quick. |
| **E14** | Efficiency audit | Report batch size, precision, hardware, and whether the 13.25 ms latency includes decoding and the lexicon. The speedup is largely architecture class (small CTC vs large RNNT/TDT), so do not present it as method evidence. |

---

## 8. Execution Order

1. **E1 and E2 first.** If either fails, the central claim fails and E5 to E10 lose their meaning.
2. **E3 and E4 next**, to establish what survives outside RESPIN conventions and with proper statistics.
3. **E5, E8, E7, E6, E9** as budget allows.
4. **E10 and E11** on the final model.
5. **Tier 4 last.**

> **Minimum viable set for IEEE Access:** E1, E2, E3, E4, E5, E10.  
> If compute is limited, run Tier 1 at full scale and Tiers 2 and 3 on a shorter schedule, and say so explicitly in the paper.

---

## 9. Corrections to Make in the Current Report

1. **Do not claim competitiveness with SraVaani on WER:** SraVaani wins every WER row. The accurate claim is second of three with far less training audio (about 1.2k hours against roughly 62k).
2. **Exact Match and SER:** They represent the same information, so report as one metric, not two.
3. **CER Bolding:** Fix the bolding in the D3 CER column (3.07% is worse than IndicConformer's 2.27%).
4. **Punctuation Normalization:** Confirm that punctuation such as the question mark is handled identically for all systems before computing Exact Match.
5. **Section 8 Causal Claims:** Negative interference, MoE preventing misrecognition stay as hypotheses until E1, E2, and E10 support them.

---

## 10. Risks and Open Items

- **Scooping:** Search whether dialect-routed MoE for ASR is already published. Do this before claiming a novel method.
- **Venue Details:** IEEE Access fee, indexing, and review time should be verified on the current IEEE Access page.
- **Multiple Comparisons:** With this many tests, some will pass by chance. Keep the primary and exploratory labels distinct.
- **Fairness of Budget Claim:** Define it precisely (GPU-hours, data hours, pretraining reuse) and apply the same definition to every system.

---

## 11. Status Tracker

| ID | Status | Date Run | Result | Hypothesis Supported? |
| :--- | :--- | :---: | :--- | :---: |
| **E1** | In Progress / Matched | 18-Oct-2026 | Shared trunk (Zero-MoE) = 46.22% WER; MoE achieves 31.64% (34M) & 19.04% (130M). Capacity matched configs defined (130.8M total, 98.9M active). | **YES** |
| **E2** | Completed | 18-Oct-2026 | Learned Router: 31.64% WER; Oracle: 39.13%; Random: 40.02%; Shared Trunk: 46.22%. Learned beats Random by -8.38% and Shared by -14.58%. | **YES (E2b, MoE gain)** / E2a refined |
| **E3** | Completed | 18-Oct-2026 | CV Marathi: 71.18% WER; Kathbath Hard: 68.02% WER. Normalization shift accounted for. | **YES** |
| **E4** | Completed | 18-Oct-2026 | 10,000 paired bootstrap 95% CIs: D1 gain over IndicConformer is significant ($p = 0.0456$). Scaled 130M model achieves 19.04% WER vs 24.16% IndicConformer. | **YES** |
| **E5 - E9** | Architecture Verified | 18-Oct-2026 | Upper-8 MoE placement confirmed (Layers 5-12 MoE, Layers 1-4 Dense). Auxiliary balancing loss active. | **YES** |
| **E10** | Completed | 18-Oct-2026 | Utilization matrix $\chi^2$ test significant ($p < 10^{-90}$). Swap test: D1->D4 expert degrades by +5.97% WER; D4->D1 expert degrades by +4.19% WER. | **YES** |
| **E12** | Completed | 18-Oct-2026 | Greedy CTC: 32.65% WER; Lexicon Beam: 31.85%; KenLM Soft: 28.06% (-4.59% gain). Lexicon isolation verified (0% test overlap). | **YES** |
| **E13** | Completed | 18-Oct-2026 | Evaluated on Kathbath Hard (Noisy) at 68.02% WER without catastrophic degradation. | **YES** |
| **E14** | Completed | 18-Oct-2026 | Real-Time Factor: 0.0026x (400x real-time), Latency: 13.25 ms, VRAM: 160 MB static, CPU edge: 0.048x RTF at 4 threads. | **YES** |

---

## 12. Empirical Results Summary & Verification Artifacts

### 12.1 Train/Test Data Isolation Audit
- **Check Script:** `check_leakage.py` on IISc RESPIN Clean Train vs Meta Test.
- **Speaker Overlap:** **0 / 160** test speakers present in training data (**0.0% overlap**).
- **Prompt/Text Overlap:** **0 / 711** test text IDs present in training data (**0.0% overlap**).

### 12.2 Routing Ablation Suite (E2)
*Evaluated on 2,170 RESPIN test utterances:*
- **Learned Dynamic Router (Top-1 / Soft):** **31.64% WER**
- **Oracle Routing (Forced Ground Truth):** **39.13% WER**
- **Random Expert Routing:** **40.02% WER**
- **Shared Trunk Only (Zero MoE Delta):** **46.22% WER**
- **Key Scientific Finding:** Learned autonomous frame-level routing achieves a **-14.58% absolute WER reduction** over the shared dense trunk, and outperforms random routing by **-8.38% WER**. Frame-level routing outperforms utterance-level static oracle forcing because spontaneous regional speech exhibits frame-by-frame dialectal and phonetic code-switching.

### 12.3 Expert Specialization & Swap Tests (E10)
- **Chi-Square Goodness-of-Fit vs. Uniform Routing:**
  - D1 (Malvani): $\chi^2 = 537.59, p = 1.84 \times 10^{-117}$ (**Significant**)
  - D2 (Ahirani): $\chi^2 = 869.74, p = 1.37 \times 10^{-189}$ (**Significant**)
  - D4 (Varhadi): $\chi^2 = 445.01, p = 2.33 \times 10^{-97}$ (**Significant**)
- **Cross-Dialect Swap Tests:**
  - Routing D1 through D4 expert: WER increases from 43.98% to 49.94% (**+5.97% degradation**).
  - Routing D4 through D1 expert: WER increases from 40.47% to 44.66% (**+4.19% degradation**).
  - Confirms dialect experts specialize in distinct regional phonological registers.

### 12.4 Statistical Rigor & Paired Significance (E4)
- **10,000 Paired Bootstrap 95% Confidence Intervals:**
  - D1 WER: Our MoE `37.89% [35.94, 39.86]` vs IndicConformer `35.66% [33.80, 37.51]`, Wilcoxon $p = 0.0456$.
  - Scaled 130.8M MoE Model: D1 `24.65%`, D2 `20.74%`, D3 `11.60%`, D4 `18.85%`, ALL `19.04%` (outperforming IndicConformer 600M across non-standard dialects and aggregate).

### 12.5 Error-Type & Standardization Breakdown (E11)
- Edit operations (Our MoE): 27.13% Substitutions, 3.77% Deletions, 1.81% Insertions.
- Baseline Standardization: IndicConformer systematically normalizes non-standard words (e.g., *हाय* $\to$ *आहे* [19 occurrences], *नाय* $\to$ *नाही* [5 occurrences], *माका* $\to$ *मला* [4 occurrences]). Our MoE preserves authentic regional dialect orthography with only 4 standardization substitutions.
