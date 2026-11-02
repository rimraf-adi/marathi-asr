---
language:
- mr
license: mit
tags:
- automatic-speech-recognition
- ctc
- conformer
- mixture-of-experts
- marathi
- speech
datasets:
- rimrafadii/respin-iisc
- ai4bharat/Kathbath
- ARTPARK-IISc/Vaani
pipeline_tag: automatic-speech-recognition
---

# Marathi Conformer-256 Sparse MoE (33M Parameters) 🎙️

An ultra-efficient, dialect-aware Automatic Speech Recognition (ASR) model tailored for Marathi and its regional dialect continuum (Malvani, Ahirani, Standard Marathi, Varhadi).

## Architecture Highlights
- **Base Architecture**: 12-layer Conformer ($d_{	ext{model}}=256$, 4 attention heads, kernel size 31).
- **Dialect Sparse Mixture-of-Experts**: 3 dedicated regional expert branches + 1 combining shared trunk.
- **Parameters**: 33.8M total parameters, with only **25.2M active parameters** per forward pass.
- **Inference Speed**: Real-Time Factor (RTFx) speedup exceeding **280x** on modern GPUs.

## Evaluation Results (mr-norm-1 Normalized)
- **RESPIN Marathi Aggregate WER**: 24.8%
  - Malvani (D1): 29.2%
  - Ahirani (D2): 25.4%
  - Standard (D3): 15.6%
  - Varhadi (D4): 24.1%
- **Kathbath Test (Clean)**: 12.8%
- **FLEURS Test**: 15.4%

## Normalization Protocol
Evaluated under the standardized `mr-norm-1` protocol (NFC unicode decomposition, Devanagari numerals `०-९` mapped to `0-9`, punctuation/dandas stripped, lowercased Latin).
