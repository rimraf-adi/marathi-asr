# [2026-10-18] Literature and Prior Art Audit: Dialect-Aware MoE for Indic / Marathi ASR

**Date:** 2026-10-18 (18 October 2026)  
**Target:** IEEE Access Submission  
**Purpose:** Pre-submission scooping verification and novelty differentiation.

---

## 1. Prior Art Landscape

### 1.1 RESPIN and MADASR Challenge (ASRU)
- **Dataset Context:** The IISc SPIRE Lab RESPIN corpus was introduced as a multi-lingual multi-dialect benchmark covering 9 Indian languages and 38+ dialects, powering the MADASR Challenge at IEEE ASRU.
- **Common Approaches in Literature:**
  - Multi-task learning with dialect identification (DID) auxiliary heads.
  - Fine-tuning monolithic foundation models (e.g. OpenAI Whisper-large-v3, IndicWav2Vec, MMS) using parameter-efficient fine-tuning (PEFT) like LoRA or MixLoRA (adapter-level MoE).

### 1.2 Dialect / Accent MoE in Global Literature
- **DialectMoE / Accent-Specific MoE (Interspeech / ACL / ICASSP):**
  - Explored primarily on regional Chinese dialects (e.g., Sichuan vs. Henan) and multi-accent English (e.g. Common Voice accents).
  - Most architectures replace FFNs entirely with pure sparse routing without a permanent shared combining trunk.

---

## 2. Our Novel Architectural Contributions & Differentiation

| Dimension | Typical Literature / Challenge Submissions | Our Work (IEEE Access Paper) |
| :--- | :--- | :--- |
| **Model Type** | Heavy monolithic foundation models (Whisper 1.5B, XLS-R 300M/1B) with LoRA | **Standalone, custom 130.8M Sparse MoE Conformer** (98.9M active params) |
| **MoE Design** | Standard Switch/GShard replacement of FFNs or LoRA adapter routing | **Shared-Trunk + Regional Dialect Expert Delta**: Always-active Standard Marathi (D3) combining trunk + dynamic additive regional expert deltas |
| **Layer Placement** | Uniform MoE across all layers or last layer only | **Upper-8 MoE placement**: Preserves dense lower 4 layers for universal acoustic/spectrogram representations |
| **Routing Ablation (E2)** | Usually only reports learned router WER | **Rigorous 4-way ablation**: Learned (31.64%) vs. Oracle (39.13%) vs. Random (40.02%) vs. Shared-only (46.22%) |
| **Swap Tests (E10)** | Rarely evaluated | **Phonological cross-dialect swap tests**: Proves D1 and D4 experts have non-interchangeable acoustic representations (+5.97% and +4.19% degradation when swapped) |
| **Efficiency** | > 100 ms latency, > 2 GB VRAM | **13.25 ms latency, 0.0026× RTF, 160 MB static VRAM** |

---

## 3. Verdict
- **Scooping Risk**: **LOW / CLEAR**.
- There is no published paper implementing a dedicated **Shared-Trunk + Routed Regional Expert Conformer** specifically isolating Marathi regional dialects (Malvani, Ahirani, Varhadi, Standard) with complete swap testing and bootstrap significance rigor.
- The paper's contribution is clearly positioned as an architectural, data-efficient, and linguistic-preservation study.
