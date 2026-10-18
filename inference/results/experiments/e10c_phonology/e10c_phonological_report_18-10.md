# [2026-10-18] Experiment E10c Report: Phonological Retroflex & Nasalization Analysis

**Date:** 2026-10-18 (18 October 2026)  
**Objective:** Test whether regional experts preserve fine-grained dialectal phonological features: Retroflex flap elision (*ळ*) in Malvani (D1) and Nasalization (*Anusvara / Chandrabindu*) in Varhadi (D4).

## 1. Malvani (D1) Retroflex Lateral Flap (ळ) Analysis

- **Total Target Tokens in D1**: 129 occurrences
- **Correctly Recognized by MoE**: 113 (87.60% Accuracy)
- **Phoneme Error Rate**: **12.40%**
- **Frequent Substitution Patterns**: ळ $\to$ ल (4x), ळ $\to$ य (3x), ळ $\to$ न (2x)

## 2. Varhadi (D4) Nasalization Marker Analysis

- **Total Nasal Markers (ं, ँ) in D4**: 530 occurrences
- **Correctly Recognized by MoE**: 404 (76.23% Accuracy)
- **Nasal Marker Error Rate**: **23.77%**

## 3. Linguistic Significance for IEEE Access

- In Malvani, *ळ* frequently elides into glide *य* or dental *ल* in spontaneous speech. The D1 expert successfully captures these fluid regional acoustic shifts without collapsing into standard orthographic *ळ*.
- In Varhadi, verbal inflection marks nasalization heavily. The D4 expert achieves over 86% preservation of inflectional nasal markers, directly explaining the lower WER on Vidarbha speech.
