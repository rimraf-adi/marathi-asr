# Journal Paper Roadmap: Multi-Dialect Streaming Conformer ASR for Marathi

**Working Title Options** (ranked by impact):
1. *"Dynamic Multi-Exit Conformer with Hierarchical Mixture-of-Experts for Multi-Dialect Streaming ASR in Marathi"*
2. *"Jointly Optimizing Latency, Accuracy, and Dialect Adaptation in Streaming ASR via GRPO-Routed Early Exit and Sparse MoE"*
3. *"A 6-Stage Curriculum for Low-Resource Multi-Dialect ASR: From Self-Supervised Pretraining to Calibrated Streaming Inference"*

**Target Venues**: IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP) · Computer Speech & Language (Elsevier) · Speech Communication (Elsevier) · INTERSPEECH (conference) · ICASSP (conference)

---

## Part 1 — What You Already Have (Core Contributions)

These are the publishable contributions that **already exist** in the codebase:

### Contribution 1: 6-Stage Curriculum Learning Pipeline for Streaming ASR

No prior work (to our knowledge) presents a fully integrated **6-stage sequential curriculum** that progressively transforms a randomly initialized Conformer through:

```
SSL → CTC Alignment → Multi-Exit → Dialect MoE → Calibration → Evaluation
```

**Why this matters**: Most papers address one or two of these stages in isolation. The *curriculum design itself* — where each stage inherits and builds upon the previous — is the contribution. The zero-leakage guarantee between streaming pretraining (Stages 1–3, HuggingFace) and local benchmark (Stages 4–6, RESPIN) is a methodological contribution that reviewers will appreciate.

**Paper angle**: Frame it as a *systems paper* demonstrating end-to-end design principles, not just one component.

---

### Contribution 2: GRPO-Routed Dynamic Early Exit for Streaming ASR

The Early-Exit Router trained via **Group Relative Policy Optimization (GRPO)** is novel in the ASR domain:

- GRPO was introduced by DeepSeek for LLM alignment (2024). Applying it to **acoustic model early-exit routing** is a novel transfer.
- The router learns to predict optimal exit depth (Layer 4/8/12) from Layer 4 representations, trading off CER vs. compute cost.
- The reward function $R = (1 - \text{CER}) - \lambda \cdot \text{Cost}(e)$ with annealed latency penalty is a clean formulation.
- No external critic network required → fits in 24GB VRAM.

**Novelty claim**: "First application of GRPO to early-exit routing in CTC-based streaming ASR."

---

### Contribution 3: Shared-Trunk + 3-Expert Dialect MoE via Sparse Upcycling

The Hierarchical MoE design with:
- A **Combining Shared FFN** (always-active Standard Marathi anchor)
- 3 **Dialect-Specific Experts** (Malvani, Ahirani, Varhadi)
- **Sparse upcycling** from the pretrained dense FFN (mathematically identical output at Step 0)
- Two-phase training: Phase 4A (hard routing by dialect metadata) → Phase 4B (soft learned routing)

This is a specific instance of the "Shared-Trunk + Routed Experts" paradigm applied to **dialectal variation in a single language**, which is underexplored. Most MoE ASR work focuses on multilingual routing (different languages), not intra-language dialect routing.

**Novelty claim**: "Intra-language dialect-aware MoE with sparse upcycling for Marathi regional variation."

---

### Contribution 4: Dynamic-Chunk Causal Training with Unified Streaming/Offline Model

The dynamic chunk masking schedule:
- $C=1$ (40ms, strictly causal): 25%
- $C=4$ (160ms): 25%
- $C=8$ (320ms): 20%
- $C=16$ (640ms): 15%
- $C=-1$ (full context): 15%

produces a **single model** that operates across all latency regimes without retraining. This is inspired by UniCAT / Unified Streaming (ICASSP 2023), but applied to a low-resource Indic language with dialect variation.

---

### Contribution 5: Dual-Head CTC Architecture (Standard + Dialectal)

Two lightweight CTC projection heads (~27K params each) on a shared backbone:
- **Head 1**: Normalized Standard Marathi (for NLP/search/banking)
- **Head 2**: Verbatim dialectal phonetic transcription (for cultural preservation)

This is a practical contribution with real-world deployment value. One forward pass, two outputs.

---

## Part 2 — Five Possible Paper Framings

### Paper A: The Full System Paper (→ TASLP / Computer Speech & Language)

**Title**: *"A 6-Stage Curriculum for Multi-Dialect Streaming ASR: Dynamic Early Exit, Sparse MoE, and GRPO Routing for Marathi"*

**Framing**: Present the full pipeline as a cohesive system. Each stage addresses a specific challenge:

| Stage | Challenge Addressed | Technical Solution |
|-------|--------------------|--------------------|
| 1 | Limited labeled data | SSL masked reconstruction on 1,435 hrs |
| 2 | Streaming constraint | Dynamic-chunk causal CTC alignment |
| 3 | Inference cost | GRPO-routed multi-exit (3x speedup) |
| 4 | Dialect variation | Sparse MoE with shared trunk |
| 5 | Reliability | Temperature calibration + GRPO polish |
| 6 | Scientific integrity | Speaker-disjoint held-out evaluation |

**Strengths**: Comprehensive; covers novel ground at each stage. Suitable for a journal where page limits are generous.

**Weakness**: Breadth over depth. Each component individually isn't as deep as a focused paper.

**Recommendation**: ⭐ **Best fit** for a journal paper (TASLP or CSL). 15-18 pages with detailed ablations.

---

### Paper B: The Early Exit + GRPO Paper (→ ICASSP / INTERSPEECH)

**Title**: *"GRPO-Routed Early Exit for Streaming Conformer ASR: Jointly Optimizing Accuracy and Latency Without a Critic Network"*

**Framing**: Focus solely on the multi-exit architecture + GRPO router. Drop the MoE and calibration discussion. Compare against:
- Fixed exit at each layer (Layer 4, 8, 12 individually)
- Confidence-based thresholding (exit when CTC entropy < threshold)
- Random routing baseline
- Oracle routing (always pick the best exit per-sample)

**Required new work**: Thorough ablation of GRPO hyperparameters ($\lambda$, $\beta_{KL}$, entropy weight, group size G), convergence curves, per-sample exit distribution analysis.

---

### Paper C: The Dialect MoE Paper (→ SLT / INTERSPEECH)

**Title**: *"Sparse Upcycling for Intra-Language Dialect Adaptation: A Shared-Trunk MoE Approach for Marathi Regional ASR"*

**Framing**: Focus on the MoE architecture. Key experiments:
- Ablate number of experts (1, 2, 3, 4, 6)
- Compare hard vs. soft vs. top-2 routing
- Show per-dialect CER breakdown (Malvani vs. Ahirani vs. Varhadi vs. Standard)
- Visualize expert specialization (what phonemes each expert learns)
- Compare against dialect-ID conditioning (embedding-based) and adapter-based approaches

---

### Paper D: The Low-Resource Indic ASR Paper (→ ACL Findings / LREC-COLING)

**Title**: *"Building a Production-Ready Streaming ASR for Marathi: Lessons from 1,435 Hours of Multi-Dialect Speech"*

**Framing**: Emphasize the **data engineering** and **low-resource challenges**: transcript sanitization, streaming from HuggingFace without local disk, handling annotation noise in Vaani, the dual-head standard/dialectal design, and the zero-leakage evaluation protocol.

This is more of an *experience paper* / *resource paper*. It would include the dataset analysis, preprocessing pipeline, and reproducibility details.

---

### Paper E: The Calibration + Deployment Paper (→ ASRU / SLT)

**Title**: *"Calibrated Multi-Exit Conformer ASR: Temperature Scaling and GRPO Sequence Polish for Reliable Dialect-Aware Streaming Inference"*

**Framing**: Focus on Stage 5 (calibration) + Stage 6 (evaluation). Show that:
- ECE drops from X% to Y% after temperature scaling
- GRPO sequence polish directly reduces CER by Z%
- Calibrated confidence scores enable reliable early-exit decisions
- Per-dialect calibration curves are different (Varhadi needs different temperature than Standard)

---

## Part 3 — Critical Improvements Needed for a Strong Paper

### 🔴 Tier 1: **Must Have** Before Submission

#### 1. Comprehensive Ablation Study

Every novel component needs an ablation. Without these, reviewers will reject immediately:

| Ablation | Experiment | What It Shows |
|----------|-----------|--------------|
| Multi-Exit vs. Single Exit | Train without multi-exit (just Layer 12) | Justifies the multi-exit architecture |
| GRPO vs. No GRPO | Train with fixed uniform routing | Justifies GRPO over random/fixed |
| GRPO vs. Confidence Threshold | Compare against entropy-based exit | GRPO vs. simpler heuristic |
| MoE vs. Dense (no experts) | Skip Stage 4 entirely | Justifies dialect experts |
| MoE: 1 vs. 2 vs. 3 experts | Vary expert count | Optimal expert count |
| Shared Trunk vs. No Shared Trunk | Remove combining FFN | Justifies the additive fusion |
| Sparse Upcycling vs. Random Init | Initialize experts randomly instead of cloning | Justifies sparse upcycling |
| Dynamic Chunks vs. Fixed Chunk | Train with single chunk size only | Justifies the chunk distribution |
| SpecAugment On vs. Off | Remove SpecAugment | Standard regularization ablation |
| With vs. Without Stage 1 SSL | Skip pretraining, start from random init at Stage 2 | Justifies SSL pretraining |
| Beam Search vs. Greedy CTC | Evaluate with and without LM decoding | Decoding contribution |
| With vs. Without Transcript Sanitization | Train on raw vs. cleaned transcripts | Data quality contribution |

**Minimum for submission**: At least 8 of the 12 ablations above.

---

#### 2. Proper Baselines and Comparisons

You need **external baselines** — your system compared against published results on the same or similar benchmarks:

| Baseline | Source | Dataset | Expected CER |
|----------|--------|---------|-------------|
| AI4Bharat IndicWav2Vec + CTC | Javed et al., 2022 | RESPIN / CommonVoice MR | ~15-20% |
| Whisper Small (fine-tuned) | OpenAI | CommonVoice MR | ~12-18% |
| Whisper Medium (zero-shot) | OpenAI | RESPIN test set | ~20-35% |
| Google USM (zero-shot) | Google | Marathi broadcast | ~10-15% |
| NVIDIA NeMo Conformer (Indic) | NVIDIA | Internal | ~12-16% |
| MMS (Meta, zero-shot) | Meta FAIR | CommonVoice MR | ~25-40% |

**Minimum**: Whisper (small + medium, zero-shot + fine-tuned), MMS (zero-shot), and at least one IndicWav2Vec baseline.

**How to get these**: Run Whisper and MMS inference on your RESPIN test set. This takes <1 hour on the A5000.

---

#### 3. Per-Dialect CER Breakdown Table

The most impactful table in any dialect-aware ASR paper:

| Model / System | D1 (Malvani) | D2 (Ahirani) | D3 (Standard) | D4 (Varhadi) | Macro CER | WER |
|----------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Whisper Small (zero-shot) | ? | ? | ? | ? | ? | ? |
| Whisper Medium (zero-shot) | ? | ? | ? | ? | ? | ? |
| **Ours: Stage 2 (CTC only)** | ? | ? | ? | ? | ? | ? |
| **Ours: Stage 3 (Multi-Exit)** | ? | ? | ? | ? | ? | ? |
| **Ours: Stage 4 (+ MoE)** | ? | ? | ? | ? | ? | ? |
| **Ours: Stage 5 (+ Calibration)** | ? | ? | ? | ? | ? | ? |
| **Ours: + Beam Search LM** | ? | ? | ? | ? | ? | ? |

This table alone can carry a paper if the MoE shows clear dialect-specific improvements.

---

#### 4. Statistical Significance Testing

- Run final evaluation 3–5 times with different random seeds (or bootstrap CER confidence intervals on the test set)
- Report mean ± std for all CER numbers
- Paired bootstrap test (or McNemar's test) for system-vs-system comparisons

---

#### 5. Latency / Compute Analysis Table

| Configuration | Avg Exit Layer | RTF (Real-Time Factor) | CER (%) | Speedup vs. Full |
|--------------|:-:|:-:|:-:|:-:|
| Layer 4 Only | 4.0 | ? | ? | ~3.0x |
| Layer 8 Only | 8.0 | ? | ? | ~1.5x |
| Layer 12 Only | 12.0 | ? | ? | 1.0x |
| GRPO Router (dynamic) | ? | ? | ? | ?x |
| Confidence Threshold (0.8) | ? | ? | ? | ?x |
| Oracle (per-sample best) | ? | ? | ? | ?x |

**Measure RTF properly**: Use `torch.cuda.synchronize()` + wall-clock timing per utterance. Average over 3 runs.

---

### 🟡 Tier 2: **Strongly Recommended** Improvements

#### 6. Scale Model to d_model=512

The current `d_model=256` (22M params) is small by modern standards. For a journal paper:
- Train a `d_model=384` (~46M params) variant for comparison
- Ideally train a `d_model=512` (~82M params) variant as the "main" result
- Show scaling behavior across 3 model sizes

This turns a single-result paper into a **scaling study**, which is much stronger.

---

#### 7. SentencePiece BPE Tokenizer (1,000–2,000 tokens)

Replace the character-level tokenizer (105 tokens) with BPE subword:
- Train SentencePiece on sanitized Shrutilipi + Vaani + IndicCorp Marathi
- 1,000–2,000 vocab → dramatically shorter CTC target sequences
- Expected CER improvement: 3–8% absolute
- **Ablation**: Compare char-105 vs. BPE-500 vs. BPE-1000 vs. BPE-2000

This is arguably the single highest-impact improvement for the final numbers.

---

#### 8. Proper Language Model Integration

Train a KenLM 5-gram on:
- IndicCorp Marathi (~550M tokens)
- Marathi Wikipedia (~25M tokens)
- Sanitized ASR transcripts

Use with `pyctcdecode` beam search (already integrated). Compare:
- Greedy CTC (current)
- Beam search without LM (lexicon-only, already built)
- Beam search + 3-gram KenLM
- Beam search + 5-gram KenLM

**Expected improvement**: 5–15% CER absolute.

---

#### 9. Longer Training Runs

Current runs are short by ASR standards:

| Stage | Current Steps | Recommended |
|-------|:---:|:---:|
| Stage 1 (SSL) | 30K | 100K–200K |
| Stage 2 (CTC) | 30K | 50K–100K |
| Stage 3 (Multi-Exit) | 25K | 30K–50K |
| Stage 4 (MoE) | 25K | 30K–50K |
| Stage 5 (Calibration) | 8K | 10K–15K |

For a journal paper, you want to show converged results, not early-stopped ones.

---

#### 10. Expert Routing Visualization

Create compelling figures:
- **Router probability heatmaps**: For each dialect (D1–D4), show the average expert routing probabilities. You want to see that Malvani speech routes to Expert 1, Ahirani to Expert 2, etc.
- **t-SNE/UMAP of expert representations**: Extract Layer 8 hidden states, color by dialect, show clustering.
- **Expert activation by phoneme**: Which Devanagari characters trigger which expert most frequently?
- **Exit layer distribution by SNR/difficulty**: Show that clean speech exits early (Layer 4) and noisy dialect speech exits late (Layer 12).

---

#### 11. Streaming Latency Evaluation Under Real Conditions

Beyond RTF, measure:
- **Endpoint latency**: Time from end of utterance to final text output
- **First-token latency**: Time from start of audio to first CTC emission
- **Chunk-size vs. CER curve**: Plot CER at each chunk size independently

This transforms the paper from "we trained a model" to "we built a deployable streaming system."

---

### 🟢 Tier 3: **Nice to Have** (Differentiators for Top Venues)

#### 12. Cross-Dialect Transfer Analysis

- Train MoE on only 2 dialects, test on unseen 3rd dialect
- Measure zero-shot dialect generalization
- Shows whether the shared trunk captures universal Marathi features

#### 13. Self-Training / Pseudo-Labeling on Untranscribed Vaani

- Use your best model to pseudo-label the untranscribed Vaani audio (the `isTranscriptionAvailable=="No"` samples)
- Re-train Stage 2 with pseudo-labels as additional data
- This is a **semi-supervised extension** that leverages the full 1,435 hrs

#### 14. Relative Positional Encoding (RoPE) Comparison

Already implemented in the codebase. Ablate:
- Absolute sinusoidal (original)
- RoPE (current)
- ALiBi (alternative)

#### 15. Conformer vs. Branchformer vs. E-Branchformer

Compare the Conformer backbone against newer architectures:
- Branchformer (Kim et al., 2022)
- E-Branchformer (Kim et al., 2023)
- Zipformer (Yao et al., 2023)

#### 16. Data Augmentation Study

Ablate:
- Speed perturbation (0.9x, 1.1x) on/off
- SpecAugment strength (weak/standard/strong)
- Noise injection on/off
- Room impulse response (RIR) simulation

---

## Part 4 — Related Work Positioning

### Must-Cite Papers (organize your Related Work around these clusters)

**Conformer Architecture**:
- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", INTERSPEECH 2020
- Kim et al., "E-Branchformer", SLT 2023

**Multi-Exit / Early Exit ASR**:
- Zhou et al., "Confident Adaptive Language Modeling", NeurIPS 2022 (CALM)
- Macoskey et al., "Multi-Exit ASR", INTERSPEECH 2021
- Peng et al., "Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context", ICML 2022

**Mixture-of-Experts in Speech**:
- You et al., "Mixture of Experts for ASR", ICASSP 2022
- Lu et al., "Scaling Speech Technology to 1,000+ Languages" (MMS), 2023
- Shen et al., "Mixture-of-Experts Meets Instruction Tuning", ICML 2024

**GRPO / RL for Sequence Models**:
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning", 2024 (introduced GRPO)
- Note: You are (likely) the first to apply GRPO to ASR routing — cite the original and claim novelty.

**Indic / Marathi ASR**:
- Javed et al., "Towards Building ASR Systems for the Next Billion Users", AAAI 2022 (IndicWav2Vec)
- Shah et al., "RESPIN: Dialect Data Collection and Benchmark", LREC 2024
- Pratap et al., "Scaling Speech Technology to 1,000+ Languages", 2024 (MMS)
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", ICML 2023 (Whisper)

**Dynamic Chunking / Streaming**:
- Wu et al., "U2: Unified Streaming and Non-Streaming Two-Pass ASR", 2020
- An et al., "UniCAT: Unified Chunk-Aware Transformer for Streaming ASR", ICASSP 2023
- Yao et al., "WeNet: Production Oriented Streaming and Non-Streaming ASR Toolkit", 2021

**Calibration**:
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
- Woodward et al., "Confidence Measures in ASR", Speech Communication 2022

**CTC Decoding / Language Models**:
- Hannun et al., "First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs", 2014 (CTC beam search)
- Zenkel et al., "Subword Language Modeling with Neural Networks", 2017

---

## Part 5 — Concrete Execution Plan for Paper-Ready Results

### Phase 1: Data & Baselines (1 week)

- [ ] Run Whisper Small + Medium (zero-shot) on RESPIN test set → per-dialect CER
- [ ] Run MMS (zero-shot) on RESPIN test set → per-dialect CER
- [ ] Train SentencePiece BPE tokenizer (1,000 tokens) on sanitized corpus
- [ ] Train KenLM 5-gram on IndicCorp + sanitized transcripts
- [ ] Compute all baseline CER/WER numbers and fill the comparison table

### Phase 2: Main Training Run (2–3 weeks)

- [ ] Train full 6-stage pipeline with BPE tokenizer + longer steps + d_model=384 or 512
- [ ] Record all per-stage metrics, convergence curves, per-dialect breakdowns
- [ ] Run beam search evaluation with KenLM after Stage 5

### Phase 3: Ablation Experiments (2 weeks, can overlap)

- [ ] Multi-exit ablation: fixed vs. GRPO vs. confidence vs. oracle
- [ ] MoE ablation: 0, 1, 2, 3 experts
- [ ] Tokenizer ablation: char-105 vs. BPE-1000
- [ ] Chunk ablation: fixed C=4 vs. fixed C=-1 vs. dynamic
- [ ] SSL ablation: with vs. without Stage 1
- [ ] Decoding ablation: greedy vs. beam vs. beam+LM

### Phase 4: Analysis & Visualization (1 week)

- [ ] Expert routing probability heatmaps per dialect
- [ ] Exit layer distribution histograms by utterance difficulty
- [ ] Convergence curves for all stages (loss + CER)
- [ ] Calibration reliability diagrams (ECE plots)
- [ ] RTF measurements with proper CUDA synchronization

### Phase 5: Writing (2–3 weeks)

- [ ] Introduction + motivation
- [ ] Related work (organized by the clusters above)
- [ ] Method: 6-stage pipeline, architecture, GRPO, MoE
- [ ] Experiments: datasets, baselines, ablations, results tables
- [ ] Analysis: routing visualization, scaling, latency
- [ ] Conclusion + limitations + future work

---

## Part 6 — Potential Weaknesses (Anticipate Reviewer Concerns)

| Likely Reviewer Concern | Pre-Emptive Response |
|------------------------|---------------------|
| "Single language — how does this generalize?" | Argue that intra-language dialect variation is an underexplored problem; the architecture is language-agnostic. |
| "Model is small (22M params)" | Scale to d_model=384/512 or frame it as an **efficient model** contribution (22M params, single GPU). |
| "No comparison with Whisper/MMS" | Must add these baselines — non-negotiable. |
| "CER numbers not competitive" | BPE tokenizer + KenLM will dramatically improve numbers. If still not SOTA, frame as "first integrated system for multi-dialect Marathi streaming" rather than SOTA chasing. |
| "GRPO router collapsed in initial run" | Fixed with entropy regularization — show that the fix works and ablate entropy weight. |
| "Only 4 Marathi dialects" | RESPIN only has 4 dialect labels. Acknowledge as limitation, suggest extension to Vaani's broader dialect coverage. |
| "No real streaming evaluation" | Add first-token latency and chunk-by-chunk evaluation metrics. |

---

## Part 7 — One-Paragraph Abstract Draft

> We present a streaming-ready Conformer ASR system for Marathi and its four major regional dialects (Malvani, Ahirani, Standard, and Varhadi), built through a novel 6-stage curriculum spanning self-supervised pretraining on 1,435 hours of multi-dialect speech, dynamic-chunk causal CTC alignment, GRPO-optimized multi-exit early inference, dialect-aware sparse Mixture-of-Experts (MoE) adaptation, temperature-calibrated confidence scoring, and frozen held-out evaluation. The architecture introduces three key innovations: (1) a **GRPO-routed early-exit mechanism** that dynamically selects inference depth (Layer 4, 8, or 12) per utterance, achieving up to 3× compute savings without an external critic network; (2) a **Shared-Trunk + 3-Expert dialect MoE** initialized via sparse upcycling from the pretrained FFN, enabling dialect-specific specialization while preserving standard Marathi as an always-active anchor; and (3) a **dual-head CTC architecture** producing both normalized standard and verbatim dialectal transcriptions in a single forward pass. Evaluated on the IISc RESPIN benchmark (2,170 utterances, 160 unseen speakers, 4 dialects), our system achieves [X]% macro-averaged CER with beam search decoding, outperforming Whisper Medium (zero-shot) by [Y]% absolute while operating at [Z]× real-time on a single GPU. Comprehensive ablations validate each pipeline stage, demonstrating that GRPO routing, MoE dialect experts, and calibrated early exit each contribute independently measurable improvements.

---

## Summary: What To Do Next

| Priority | Action | Impact on Paper |
|----------|--------|----------------|
| 🔴 **P0** | Run Whisper + MMS baselines on RESPIN test | Cannot submit without external baselines |
| 🔴 **P0** | Complete full ablation table (at least 8 ablations) | Reviewers will desk-reject without ablations |
| 🔴 **P0** | Per-dialect CER breakdown table | The central result of a dialect-aware paper |
| 🟡 **P1** | Train BPE-1000 tokenizer + retrain pipeline | Likely 3-8% CER improvement |
| 🟡 **P1** | Train KenLM 5-gram + beam search eval | Likely 5-15% CER improvement |
| 🟡 **P1** | Scale to d_model=384 or 512 | Competitive parameter count |
| 🟢 **P2** | Expert routing visualizations | Strong qualitative evidence |
| 🟢 **P2** | Proper streaming latency benchmarks | Deployment credibility |
| 🟢 **P2** | Confidence interval / bootstrap testing | Statistical rigor |
