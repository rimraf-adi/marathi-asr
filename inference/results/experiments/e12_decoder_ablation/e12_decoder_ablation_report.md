# [2026-10-18] Experiment E12 Report: Decoder Ablation & Lexicon Isolation Audit

**Date:** 2026-10-18 (18 October 2026)  
**Test Set:** IISc RESPIN Held-Out Marathi (`meta_test_mr.json`, 2,170 utterances, 3.041 hours)  
**Evaluated Decoders:** Greedy CTC, Lexicon-Only Beam Search, KenLM 5-Gram ($\alpha=0.5$), KenLM Soft ($\alpha=0.15$)  

---

## 1. Lexicon Construction & Text Overlap Audit (Ground Rule Requirement)

> [!IMPORTANT]
> **Plan Question:** *Does the lexicon include RESPIN text? If the lexicon helps this model far more than the baselines, check for text overlap.*
>
> **Audit Finding:**
> - Source code inspected in [`decoding/beam_search_decoder.py`](file:///d:/marathi-asr/decoding/beam_search_decoder.py#L138-L144).
> - The 30,000-word unigram lexicon is mined **strictly from the `train` and `calibration` splits** of `respin_split_cache.json`.
> - The held-out test split was **never seen** during wordlist construction.
> - Train/Test isolation verified via `check_leakage.py`: **0 / 711 text ID overlap (0.0% prompt leakage)**.

---

## 2. Performance Comparison Across Decoders (Aggregate ALL)

| Decoder Configuration | Beam Width | LM Weight ($\alpha$) | Word Insertion ($\beta$) | CER (%) | WER (%) | SER (%) | Exact Match (%) | Latency (ms/utt) | RTF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy CTC Baseline** | N/A | 0.0 | 0.0 | 7.97% | 32.65% | 83.82% | 16.18% | **69.0 ms** | **0.0137x** |
| **Lexicon Beam Search** | 16 | 0.0 | 0.0 | 7.81% | 31.85% | 83.55% | 16.45% | 85.8 ms | 0.0170x |
| **KenLM 5-gram (Default)**| 16 | 0.5 | 1.5 | 7.23% | 28.29% | 78.11% | 21.89% | 81.0 ms | 0.0161x |
| **KenLM 5-gram (Soft)** | 16 | 0.15 | 1.0 | **6.78%** | **28.06%** | **77.28%** | **22.72%** | 82.9 ms | 0.0164x |

---

## 3. Dialect-by-Dialect Breakdown (WER %)

| Dialect | Name | Greedy CTC | Lexicon Beam | KenLM Default ($\alpha=0.5$) | KenLM Soft ($\alpha=0.15$) | LM Gain (Soft vs. Greedy) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | Malvani / Konkan | 39.38% | 38.45% | 39.38% | 37.75% | **-1.63%** |
| **D2** | Ahirani / Khandesh | 34.11% | 33.46% | 30.90% | 31.05% | **-3.06%** |
| **D3** | Standard Marathi | 22.89% | 22.24% | 15.17% | 16.15% | **-6.74%** |
| **D4** | Varhadi / Vidarbha | 33.71% | 32.74% | 26.95% | 26.56% | **-7.15%** |
| **ALL** | **Aggregate Average** | 32.65% | 31.85% | 28.29% | **28.06%** | **-4.59%** |

---

## 4. Key Scientific Insights

1. **Standard LM Penalty on Regional Dialects**:
   - Applying standard 5-gram KenLM with standard weight ($\alpha=0.5$) significantly helps Standard Marathi (**-7.72% WER**) but yields **zero gain on Malvani D1 (0.00% delta)**.
   - Because standard KenLM n-gram models assign lower probabilities to non-standard dialect syntax and regional lexical tokens (e.g. *हाय*, *नाय*, *माका*), standard LM decoding acts as a normative regularizer that counteracts dialect preservation.
2. **Soft LM Constraint ($\alpha=0.15$)**:
   - Softening the language model weight ($\alpha=0.15$) resolves this conflict, providing gains across all dialects including D1 (**-1.63% WER**) and achieving the best aggregate WER (**28.06%**).
3. **Lexicon-Only Contribution**:
   - Constraining the search space to the training vocabulary alone without LM scoring yields a modest **-0.80% WER reduction**, proving that the acoustic MoE model carries the dominant recognition capacity.
