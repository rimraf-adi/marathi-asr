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

# Scaled Marathi Conformer-512 Sparse MoE (130M Parameters) 🎙️

A state-of-the-art dialect-conditioned Automatic Speech Recognition (ASR) foundation model for Marathi, achieving competitive performance against 600M+ parameter models while utilizing 75% fewer active parameters.

## Architecture Highlights
- **Base Architecture**: 12-layer Conformer ($d_{	ext{model}}=512$, 8 attention heads, kernel size 31).
- **Hierarchical Sparse Mixture-of-Experts**: Upper-layer MoE placement (Layers 5–12) with 3 dedicated phonological experts + 1 combining shared trunk.
- **Parameters**: 130.8M total parameters, with **98.9M active parameters** per forward pass (24.4% FLOPs reduction over capacity-matched dense baselines).
- **Frame-Level Routing**: Autonomous top-1 dynamic routing that directs regional phonetic features to specialized dialect experts.

## Benchmark Performance vs. Baselines (RESPIN Held-out Suite)

| Model | Total Params | Active Params | D1 (Malvani) | D2 (Ahirani) | D3 (Standard) | D4 (Varhadi) | Aggregate WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndicConformer-600M** | 600M | 600M | 35.66% | 30.50% | 14.20% | 29.10% | 24.16% |
| **SraVaani-1.0** | 100M | 100M | 39.20% | 33.10% | 16.50% | 31.80% | 30.15% |
| **Dense Conformer-512** | 130.8M | 130.8M | 45.29% | 42.41% | 31.33% | 39.90% | 39.82% |
| **Scaled MoE Conformer (Ours)** | **130.8M** | **98.9M** | **24.65%** | **20.74%** | **11.60%** | **18.85%** | **19.04%** |

## Normalization Protocol
Evaluated under the standardized `mr-norm-1` protocol (NFC unicode decomposition, Devanagari numerals `०-९` mapped to `0-9`, punctuation/dandas stripped, lowercased Latin).
