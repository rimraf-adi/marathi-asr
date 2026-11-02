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
pipeline_tag: automatic-speech-recognition
---

# Dense Marathi Conformer-Small (Dense Trunk Baseline) 🎙️

A lightweight 12-layer dense Conformer backbone trained under self-supervised pretraining (117,000 steps) on Marathi speech.

## Architecture Highlights
- **Base Architecture**: 12-layer Dense Conformer ($d_{	ext{model}}=256$, 4 attention heads, kernel size 31).
- **Parameters**: 20.5M parameters (100% active).
- **RESPIN Aggregate WER**: 46.22%
