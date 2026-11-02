---
language:
- mr
license: mit
tags:
- automatic-speech-recognition
- ctc
- conformer
- dense
- marathi
- speech
datasets:
- rimrafadii/respin-iisc
- ai4bharat/Kathbath
- ARTPARK-IISc/Vaani
pipeline_tag: automatic-speech-recognition
---

# Capacity-Matched Dense Marathi Conformer-512 (130M Parameters) 🎙️

A fully dense, capacity-matched Conformer foundation model for Marathi ASR, trained across 300,000 alignment steps to serve as the strict parameter-matched control (130.8M total parameters) against the dialect-conditioned Sparse MoE Conformer.

## Architecture Highlights
- **Base Architecture**: 12-layer Dense Conformer ($d_{	ext{model}}=512$, 8 attention heads, FFN expansion factor = 8, kernel size 31).
- **Parameters**: 130,812,425 parameters (100% active per forward pass).
- **Decoder**: CTC greedy and beam decoding over 105 character vocabulary.

## Benchmark Performance (RESPIN Held-out Suite)
- **Malvani (D1)**: 45.29%
- **Ahirani (D2)**: 42.41%
- **Standard (D3)**: 31.33%
- **Varhadi (D4)**: 39.90%
- **Aggregate WER**: **39.82%** (CER: 11.40%)
