# [2026-10-18] IISc RESPIN Marathi ASR Benchmark: Rigorous Model Comparison Report

- **Evaluated Test Set**: IISc RESPIN Held-Out Marathi (`meta_test_mr.json`)
- **Total Utterances**: 2,170 (3.041 hours)
- **Evaluation Protocol**: Zero-leakage held-out test evaluation using **16-Beam Width Decoding + Lexicon**
- **Target Dialects**: D1 (Malvani/Konkan), D2 (Ahirani/Khandesh), D3 (Standard Marathi), D4 (Varhadi/Vidarbha)
- **Normalization**: Unified text normalization stripping terminal punctuation (e.g. `?`, `!`, `।`), diacritics, and whitespace uniformly across all references and hypotheses.

---

## 1. Overall System Performance Summary (Aggregate ALL)

> [!NOTE]
> **Data Efficiency Context**: SraVaani 1.0 is trained on ~62,000 hours of multilingual/Indic audio and achieves the lowest overall WER. Our MoE Conformer is trained on ~1,200 hours (~50× less data) and ranks 2nd overall, outperforming AI4Bharat IndicConformer-600M across all non-standard dialects with a 4.5× smaller active parameter footprint.

| Model Name | Decoding | Training Data | Total Params | Active Params | CER (%) | WER (%) | SER / 1-EM (%) | RTF | Latency (ms/utt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARTPARK-IISc SraVaani 1.0** | TDT / CTC | ~62,000 hrs | 430M | 430M | **2.97%** | **15.75%** | **58.80%** | 0.0323x | 163.2 ms |
| **Our Model (Scaled MoE Conformer)** | Beam 16 | ~1,200 hrs | 130.8M | **98.9M** | 4.53% | 19.04% | 58.53% | **0.0026x** | **13.25 ms** |
| **AI4Bharat IndicConformer 600M** | CTC | ~10,000 hrs | 600M | 600M | 4.95% | 23.94% | 70.00% | 0.0693x | 349.7 ms |

---

## 2. Dialect-by-Dialect Comparative Breakdown

### Character Error Rate (CER %) by Dialect

| Dialect | Name | Our Model (Scaled MoE) | ARTPARK-IISc SraVaani 1.0 | AI4Bharat IndicConformer 600M | Our vs. IndicConformer (ΔCER) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D1** | Malvani / Konkan | 5.91% | **4.53%** | 7.39% | **-1.48% (Better)** |
| **D2** | Ahirani / Khandesh | 4.46% | **2.95%** | 4.96% | **-0.50% (Better)** |
| **D3** | Standard Marathi | 3.07% | **1.69%** | 2.27% | +0.80% |
| **D4** | Varhadi / Vidarbha | 4.59% | **2.58%** | 4.96% | **-0.37% (Better)** |
| **ALL** | **Aggregate Average** | 4.53% | **2.97%** | 4.95% | **-0.42% (Better)** |

*Note: SraVaani 1.0 achieves the lowest CER across all dialects. Our model achieves lower CER than IndicConformer 600M on D1, D2, and D4.*

### Word Error Rate (WER %) by Dialect

| Dialect | Name | Our Model (Scaled MoE) | ARTPARK-IISc SraVaani 1.0 | AI4Bharat IndicConformer 600M | Our vs. IndicConformer (ΔWER) | Relative Error Reduction (vs Indic) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | Malvani / Konkan | 24.65% | **21.71%** | 34.59% | **-9.94%** | **28.7% RER** |
| **D2** | Ahirani / Khandesh | 20.74% | **17.76%** | 24.64% | **-3.90%** | **15.8% RER** |
| **D3** | Standard Marathi | 11.60% | **8.97%** | 11.57% | +0.03% | Parity (~0.0% RER) |
| **D4** | Varhadi / Vidarbha | 18.85% | **14.14%** | 24.29% | **-5.44%** | **22.4% RER** |
| **ALL** | **Aggregate Average** | 19.04% | **15.75%** | 23.94% | **-4.90%** | **20.5% RER** |

---

## 3. Qualitative Comparative Samples (Normalized Output)

| Dialect | Ground Truth Reference | Our Model (Scaled MoE) | ARTPARK-IISc SraVaani 1.0 | AI4Bharat IndicConformer 600M | Error Mode in IndicConformer |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **D1** | `उपकरणे खैयची खैयची आसत` | `उपकरणे खैयची खैयची आसत` | `उपकरणे खैयची खैयची आसत` | `उपकरणे काहीची काहीची असतात` | Normative dialect standardization |
| **D4** | `माया खात्यात कितीक पैसे हाय हे मले कस पायता येईन` | `माया खात्यात कितीक पैसे हाय हे मले कस पायता येईन` | `माया खात्यात कितीक पैसे हाय हे मले कस पायता येईन` | `माझ्या खात्यात किती पैसे आहेत हे मला कसे पाहता येईल` | Normative dialect standardization |
| **D3** | `बकरी किंवा मेंढीपासून खत मिळते का` | `बकरी किंवा मेंढीपासून खत मिळते का` | `बकरी किंवा मेंढीपासून खत मिळते का` | `बकरी किंवा मेंढीपासून खत मिळते का` | Exact match across all systems |
| **D3** | `ठिबक सिंचनाची जोडणी कशी असावी` | `ठिबक सिंचनाची जोडणी कशी असावी` | `ठिबक सिंचनाची जोडणी कशी असावी` | `ठिबक सिंचनाची जोडणी कशी असावी` | Exact match across all systems |
| **D3** | `के वाय सी फॉर्म भरणे गरजेचे असते का` | `के वाय सी फॉर्म भरणे गरजेचे असते का` | `केवायसी फॉर्म भरणे गरजेचे असते का` | `के वाय सी फॉर्म भरणे गरजेचे असते का` | Formatting difference |
| **D3** | `मुदत खाते ऑनलाइन उघडता येईल का` | `मुदत खाते ऑनलाइन उघडता येईल का` | `मुदत खाते ऑनलाइन उघडता येईल का` | `मुदत खाते ऑनलाइन उघडता येईल का` | Exact match across all systems |

---

## 4. Architectural Hypotheses and Empirical Evidence

1. **Negative Interference Mitigation (Supported by E2 & E10)**:
   - When experts are disabled (Shared Trunk Only), aggregate WER degrades from 31.64% to 46.22% (+14.58% penalty).
   - Swapping D1 into D4 expert degrades D1 by +5.97% WER; swapping D4 into D1 degrades D4 by +4.19% WER.
   - This empirically confirms that isolated dialect experts protect non-standard phonological varieties from catastrophic gradient interference from dominant standard Marathi.

2. **Dialect Preservation vs. Normative Standardization (Supported by E11)**:
   - IndicConformer exhibits 34 systematic standardization substitutions (e.g. *हाय* $\to$ *आहे*, *माका* $\to$ *मला*), whereas our MoE exhibits only 4.
   - The MoE routing dynamically preserves authentic regional phonology without imposing standardized forms.
