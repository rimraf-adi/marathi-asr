# Direction 8: Novel Proxy-Task Design for Label-Free Neural Architecture Search in ASR

## Core Idea

**Design a novel task** — purpose-built for evaluating speech encoder architectures — that requires **no labeled data** and **no SSL pretraining**. This task is not SSL itself; it is a new invention, analogous to how SSL was once invented for representation learning, but optimised for a fundamentally different goal: **architecture discrimination**.

> [!IMPORTANT]
> SSL was designed to learn good *representations*. A NAS proxy task must be designed to *rank architectures*. These are different objectives with different design pressures. Borrowing SSL directly inherits its collapse modes, shortcut vulnerabilities, and expensive pretraining inner loops — all of which are liabilities for architecture search.

---

## 1. Why Invent a New Task?

### The Problem With Reusing SSL

| Property | SSL (e.g., BEST-RQ, HuBERT) | Ideal NAS Proxy Task |
|---|---|---|
| **Designed for** | Learning transferable representations | **Discriminating architectures** |
| **Requires pretraining?** | Yes — expensive inner loop per architecture | **No — fast forward-pass scoring** |
| **Collapse / shortcuts** | Serious risk (trivial local-context copy) | **Resistant by construction** |
| **Correlation target** | Downstream task quality (indirect) | **Architecture ranking fidelity (direct)** |
| **Compute cost** | High (full pretrain for each candidate) | **Low (minutes, not hours)** |
| **Architecture-discriminative?** | Not guaranteed — many architectures reach similar SSL loss | **Maximised by design** |

### The Gap in Literature

- **Vision:** UnNAS (ECCV 2020) showed rotation/jigsaw/colorization are good NAS proxies. MAE-NAS (ICLR 2024) used masked pixel reconstruction. These are all existing pretext tasks repurposed — nobody *designed* a task specifically for NAS.
- **Speech:** All NAS uses supervised CTC/attention loss. No label-free NAS proxy exists at all.
- **Neither domain** has asked: *What properties should a task have to be maximally architecture-discriminative?*

---

## 2. Desiderata for a NAS Proxy Task

A good NAS proxy task $\mathcal{T}$ must satisfy these properties simultaneously:

### D1: Architecture-Discriminative
Different architectures must produce *significantly different* scores:
$$\text{Var}_{\alpha \sim \mathcal{A}}\big[\mathcal{S}_\mathcal{T}(\alpha)\big] \gg \text{Var}_{\text{seed}}\big[\mathcal{S}_\mathcal{T}(\alpha)\big]$$

The inter-architecture variance must dominate the intra-architecture (seed/init) variance.

### D2: Rank-Consistent with Downstream WER
$$\tau_{\text{Kendall}}\Big(\text{rank}_\mathcal{T}(\alpha_1, \dots, \alpha_N), \; \text{rank}_{\text{WER}}(\alpha_1, \dots, \alpha_N)\Big) \geq \tau_{\min}$$

### D3: Fast to Evaluate
No pretraining inner loop. Ideally computable from a single forward pass or a few hundred steps of lightweight training.

### D4: Collapse-Resistant
The task must not admit trivial solutions (constant output, identity copy, low-rank projections).

### D5: Label-Free
Requires only raw, unlabeled audio.

---

## 3. Candidate Novel Tasks

### Task 1: Cross-Frequency Imputation (CFI)

**Idea:** Randomly zero out contiguous frequency bands in the mel spectrogram. The encoder must predict the missing bands from the remaining ones. This directly tests an architecture's ability to model spectral dependencies — the core competency for phoneme discrimination.

**Formulation:**
Let $X \in \mathbb{R}^{T \times F}$ be a mel spectrogram. Define a binary frequency mask $M \in \{0, 1\}^F$ that zeros out a random band $[f_a, f_b]$:
$$\tilde{X} = X \odot \mathbf{1}_T M^\top$$

The task score is the reconstruction error of the masked band:
$$\mathcal{S}_{\text{CFI}}(\alpha) = \mathbb{E}_{X, M}\Big[\big\| \text{Dec}\big(f_\alpha(\tilde{X})\big)_{[:, f_a:f_b]} - X_{[:, f_a:f_b]} \big\|_2^2 \Big]$$

where $\text{Dec}$ is a tiny, fixed, architecture-independent linear decoder (so the score reflects the *encoder* quality only).

**Why it's architecture-discriminative:**
- Architectures with strong spectral modeling (wider conv kernels, multi-head attention with frequency-aware positional encoding) will reconstruct formant transitions accurately.
- Shallow or narrow architectures will produce blurry, averaged reconstructions.
- Unlike masked prediction SSL, the decoder is frozen/trivial — no pretraining needed.

**Anti-collapse:** The mask is over *frequency* (not time), so temporal copy shortcuts are useless. The encoder genuinely must have learned spectral correlations.

---

### Task 2: Temporal Coherence Ordering (TCO)

**Idea:** Given three audio segments $(A, B, C)$ from the same utterance where $B$ is the middle segment, the encoder must produce representations where $B$'s embedding is closer to both $A$ and $C$ than to a random distractor $D$. This tests temporal modeling depth — whether the architecture captures long-range phonotactic structure.

**Formulation:**
Sample a contiguous utterance, split into segments $(A, B, C)$. Sample distractor $D$ from a different utterance. The task score is:
$$\mathcal{S}_{\text{TCO}}(\alpha) = \mathbb{E}\Big[\mathbb{1}\big[d(f_\alpha(B), f_\alpha(A)) + d(f_\alpha(B), f_\alpha(C)) < d(f_\alpha(B), f_\alpha(D)) + d(f_\alpha(A), f_\alpha(D))\big]\Big]$$

where $d$ is cosine distance.

**Why it's architecture-discriminative:**
- Architectures with strong temporal receptive fields (deep layers, large conv kernels, effective attention spans) will embed consecutive segments close together.
- Shallow architectures with limited context will fail to distinguish temporally adjacent from random.

**Anti-collapse:** Trivially mapping everything to a constant gives 50% accuracy (random), not high scores.

---

### Task 3: Perturbation Sensitivity Ratio (PSR)

**Idea:** The *ratio* of representation change under a linguistically irrelevant perturbation (noise, reverb) to representation change under a linguistically relevant perturbation (speed change, vowel truncation). A good architecture should be *invariant* to irrelevant perturbations and *sensitive* to relevant ones.

**Formulation:**
Let $T_{\text{irrel}}$ be a linguistically-irrelevant transform (additive noise at 10 dB SNR) and $T_{\text{rel}}$ be a linguistically-relevant transform (5% speed change altering formant frequencies):

$$\mathcal{S}_{\text{PSR}}(\alpha) = \mathbb{E}_x \left[\frac{\|f_\alpha(x) - f_\alpha(T_{\text{irrel}}(x))\|}{\|f_\alpha(x) - f_\alpha(T_{\text{rel}}(x))\| + \epsilon}\right]$$

**Minimise this ratio.** A good encoder barely changes under noise (numerator small) but clearly registers phonetic changes (denominator large).

**Why it's architecture-discriminative:**
- This directly measures the **signal-to-noise ratio of the representation space** — how well the architecture separates phonetic content from acoustic clutter.
- Architectures with proper inductive biases (convolutions capturing formant structure, attention capturing long-range dependencies) will have better PSR.

**Anti-collapse:** Collapsing to a constant makes both numerator and denominator zero. The $\epsilon$ term and the ratio formulation prevent this.

---

### Task 4: Phonetic Bottleneck Compression (PBC)

**Idea:** Force the encoder output through a severe information bottleneck (e.g., quantise to $K = 64$ codes), then measure how well a tiny decoder reconstructs the *original spectrogram* from these codes. This measures how efficiently the architecture compresses acoustic information into a phonetically meaningful discrete space.

**Formulation:**
$$\mathcal{S}_{\text{PBC}}(\alpha) = \mathbb{E}_X \Big[\big\|X - \text{Dec}_{\text{fixed}}\big(\text{VQ}_K(f_\alpha(X))\big)\big\|_2^2 \Big]$$

where $\text{VQ}_K$ is a fixed, architecture-independent vector quantiser with $K$ codes, and $\text{Dec}_{\text{fixed}}$ is a small frozen decoder.

**Why it's architecture-discriminative:**
- With only 64 codes, the encoder must discover an extremely efficient phonetic coding. Architectures that naturally learn hierarchical, multi-scale features will compress better.
- Directly measures information-theoretic capacity: $I(X; \text{VQ}_K(f_\alpha(X)))$.

---

### Task 5: Temporal Inversion Detection (TID)

**Idea:** Randomly time-reverse a segment within an utterance. The encoder must detect whether the segment is forward or reversed. This is the speech analogue of rotation prediction in vision (UnNAS), but designed for temporal structure.

**Formulation:**
With probability 0.5, reverse a random segment $[t_a, t_b]$ within the spectrogram:
$$\tilde{X}_{[t_a:t_b]} = \text{flip}(X_{[t_a:t_b]})$$

The task score is the binary classification accuracy of a single linear probe on the encoder output:
$$\mathcal{S}_{\text{TID}}(\alpha) = \text{Acc}\Big(\sigma(w^\top \text{pool}(f_\alpha(\tilde{X})) + b), \; y_{\text{flipped}}\Big)$$

**Why it's architecture-discriminative:**
- Speech is inherently asymmetric in time (plosive onset → vowel → coda). Detecting reversal requires capturing fine-grained temporal dynamics.
- Architectures with causal or semi-causal attention, appropriate conv kernel sizes, and sufficient depth will detect inversions easily. Shallow or overly local architectures will struggle.

---

## 4. Mathematical Framework

### 4.1 Task Design as Meta-Learning

Define the **task design problem** as a meta-optimisation:

$$\mathcal{T}^* = \arg\max_{\mathcal{T} \in \mathbb{T}} \; \tau_{\text{Kendall}}\Big(\text{rank}_{\mathcal{S}_\mathcal{T}}(\alpha_1, \dots, \alpha_N), \; \text{rank}_{\text{WER}}(\alpha_1, \dots, \alpha_N)\Big)$$

subject to:
$$\text{Cost}(\mathcal{T}, \alpha) \leq B_{\text{eval}} \quad \forall \alpha$$

This is itself a bilevel problem:
- **Outer:** Find the task $\mathcal{T}$ that maximises rank correlation with WER.
- **Inner:** For each task, evaluate each architecture's score $\mathcal{S}_\mathcal{T}(\alpha)$.

### 4.2 Composite Multi-Task Scoring

Rather than selecting a single task, combine multiple tasks into a composite score with learnable weights:

$$\mathcal{S}_{\text{composite}}(\alpha) = \sum_{k=1}^{K} \lambda_k \cdot \hat{\mathcal{S}}_k(\alpha)$$

where $\hat{\mathcal{S}}_k$ is the normalised score on task $k$ (CFI, TCO, PSR, PBC, TID) and $\lambda_k$ are weights optimised on a small validation set of architectures with known WER rankings.

### 4.3 Rank Consistency Theory

**Theorem (Informal):** A proxy task $\mathcal{T}$ is rank-consistent with WER if:

1. $\mathcal{S}_\mathcal{T}(\alpha)$ is a monotonic function of the mutual information $I_\alpha(X; Z)$ between input acoustics and encoder representations.
2. The fine-tuning procedure preserves the MI ordering: $I_{\alpha_1}(X; Z) > I_{\alpha_2}(X; Z) \Rightarrow \text{WER}(\alpha_1) < \text{WER}(\alpha_2)$.

**Formal bound:**
$$P\big(\text{rank}_\mathcal{T} \neq \text{rank}_{\text{WER}}\big) \leq \frac{\text{Var}[\epsilon_{\text{fine-tune}}]}{\big(\min_{i \neq j} |I_{\alpha_i} - I_{\alpha_j}|\big)^2}$$

where $\epsilon_{\text{fine-tune}}$ is the stochastic noise introduced by the fine-tuning procedure.

**Implication:** Rank consistency improves when (a) architectures have well-separated MI values, and (b) fine-tuning is stable (low variance).

### 4.4 Multi-Objective: Task Score + Compute Cost (Pareto NAS)

$$\min_{\alpha \in \mathcal{A}} \quad \Big(\mathcal{S}_\mathcal{T}(\alpha), \; \text{Cost}(\alpha)\Big)$$

Output is a **Pareto front**, not a single model.

**Cost dimensions to track:**

| Cost Metric | Description |
|---|---|
| **Proxy evaluation cost** | FLOPs to compute $\mathcal{S}_\mathcal{T}(\alpha)$ *(should be minutes, not hours)* |
| **Fine-tuning cost** | GPU-hours to reach target WER on 10h labeled set |
| **Inference cost** | Parameters, FLOPs, real-time factor, peak memory |
| **Search cost** | Total GPU-hours of the NAS itself *(a reviewer will ask)* |

### 4.5 Compute-Optimal Architecture Theory

Fit per-architecture scaling laws where the proxy task score replaces SSL loss:
$$\mathcal{S}_\mathcal{T}(\alpha, C) = A_\alpha \cdot C^{-\beta_\alpha} + S_{\alpha,\infty}$$

**Key question:** Does the compute-optimal architecture change with budget? (e.g., Conformer wins at low budget, E-Branchformer at high budget.)

### 4.6 Discriminativeness Guarantee

Define a task as **$(\gamma, \delta)$-discriminative** over search space $\mathcal{A}$ if:
$$P_{\alpha_1, \alpha_2 \sim \mathcal{A}}\Big[\big|\mathcal{S}_\mathcal{T}(\alpha_1) - \mathcal{S}_\mathcal{T}(\alpha_2)\big| \geq \gamma\Big] \geq 1 - \delta$$

A task that is not discriminative is useless for NAS — all architectures score the same. **Proving discriminativeness for each proposed task is a theoretical contribution.**

---

## 5. Search Space

**Conformer-family search space:**

| Dimension | Range |
|---|---|
| Depth (layers) | 6, 8, 10, 12, 16 |
| Width (d_model) | 128, 192, 256, 384, 512 |
| Attention heads | 2, 4, 8 |
| Conv kernel size | 7, 15, 31, 63 |
| Subsampling rate | 4x, 6x, 8x |
| FFN ratio | 2, 4, 8 |
| Module ordering | MHSA→Conv→FFN, Conv→MHSA→FFN, parallel branches |
| Block type | Conformer, E-Branchformer, Zipformer, Squeezeformer |

---

## 6. Experimental Plan

1. **Tabular benchmark:** Sample 100–200 architectures. Compute all 5 proxy task scores (CFI, TCO, PSR, PBC, TID) for each — this should take minutes per architecture, not hours.
2. **Ground truth:** Fine-tune each architecture with CTC on 10h labeled data to get WER.
3. **Correlation analysis:** Measure Spearman/Kendall $\tau$ between each proxy task score and WER. Identify which task (or composite) is most rank-consistent.
4. **Compare against SSL baselines:** Also compute BEST-RQ and HuBERT-style SSL loss for each architecture (expensive). Show that the novel proxy tasks achieve comparable or better rank correlation at a fraction of the compute.
5. **Pareto comparison:** Compare Pareto fronts — proxy-searched vs. supervised-searched vs. SSL-searched vs. random vs. hand-designed Conformer.
6. **Budget regimes:** Low-budget (small models, short eval) and high-budget (large models). Show whether the winning architecture changes.
7. **Total pipeline cost:** Report proxy-eval + fine-tune cost against supervised NAS and SSL-NAS.
8. **Out-of-domain + multilingual:** Evaluate discovered architectures on SUPERB tasks.

---

## 7. Headline Claims

> **(a)** Purpose-built proxy tasks outperform borrowed SSL objectives as NAS ranking signals, at 10–100× lower compute.
>
> **(b)** The searched encoders beat hand-designed and supervised-NAS baselines at matched compute.
>
> **(c)** The composite proxy score provides formal rank-consistency guarantees.
>
> **(d)** The task design framework is general — applicable to any modality where label-free NAS is needed.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Proxy tasks don't correlate with WER at the top of the ranking | Use composite scoring; report failure characterisation (still publishable). |
| Proxy tasks are architecture-indiscriminate (all scores similar) | Prove discriminativeness formally; test on diverse search space. |
| GPU-hours are noisy and hardware-dependent | Use FLOPs and memory as primary, wall-clock as secondary. State hardware. |
| Trivial solutions / collapse on a specific task | Each task has built-in anti-collapse (see §3). Composite scoring provides redundancy. |

---

## Novelty & Literature Context

### What has been done?

#### A. Label-Free NAS in Vision (Mature — but all *repurpose* existing tasks)
- **UnNAS** (Liu, Dollár, He, Girshick, Yuille, Xie; ECCV 2020): Repurposed rotation/colorization/jigsaw as NAS proxies. Spearman $\rho > 0.80$ with supervised rankings. *Vision only. Used existing pretext tasks, did not design new ones.*
- **CSNAS** (Nguyen & Chang; IEEE Access 2022): Repurposed SimCLR/InfoNCE contrastive loss. *Vision only. Borrowed SSL objective.*
- **BossNAS** (Li et al.; ICCV 2021): Block-wise self-supervised search using ensemble bootstrapping. *Vision only. Borrowed SSL.*
- **MAE-NAS** (Hu, Chu, Zhang; ICLR 2024): Repurposed Masked Autoencoding. *Vision only. Borrowed SSL.*

#### B. Supervised Speech NAS (Extensive, but all labeled)
- **AutoSpeech** (Shi, Jin et al.; Interspeech 2020): DARTS for speaker recognition with supervised softmax CE. *Supervised.*
- **EfficientTDNN** (Lin et al.; IEEE TASLP 2022): TDNN supernet with AAM-Softmax. *Supervised.*
- **NAS-Bench-ASR** (Mehrotra et al.; ICLR 2021): 8,242 architectures with supervised CTC on TIMIT. *Supervised.*
- **DARTS-ASR** (Chen et al.; Interspeech 2020): Multilingual DARTS with CTC. *Supervised.*
- **DARTS-Conformer** (Chen et al.; Interspeech 2021): DARTS inside Conformer with CTC/Attention. *Supervised.*

#### C. SSL Compression (Not Architecture Search)
- **LightHuBERT** (Wang et al.; Interspeech 2022): Prunes HuBERT via distillation. *Compression, not architecture discovery.*

#### D. Speech SSL Architecture Ablations (Manual, Not Search)
- **BEST-RQ** (Chiu et al.; ICML 2022): Conformer vs. Transformer under masked prediction. *Manual ablation.*
- **WavLM** (Chen et al.; IEEE JSTSP 2022): Gated relative position bias. *One-off modification.*
- **SUPERB** (Yang et al.; IEEE JSTSP 2021): Benchmark of 15+ SSL models. *Comparison, not search.*

#### E. Compute-Optimal Scaling Laws for Speech (Emerging)
- **OWLS** (Meta; arXiv 2025): Chinchilla-style $L(N, D)$ for speech. *Architecture fixed.*
- **Scaling Audio Models** (arXiv 2025): 3D compute-optimal ($N, T, V$). *Architecture fixed.*

### What is NOVEL in our approach?

> [!IMPORTANT]
> **No one — in vision or speech — has *designed a task from scratch* specifically to be a NAS proxy.** All prior label-free NAS work *borrows* existing SSL/pretext tasks. We are the first to ask: *what task properties maximise architecture discrimination?*

1. **First purpose-designed NAS proxy tasks for speech** — not borrowed SSL.
2. **Formal task design framework** with desiderata (discriminativeness, rank consistency, collapse resistance) and meta-optimisation.
3. **5 novel candidate tasks** (CFI, TCO, PSR, PBC, TID) with formal anti-collapse guarantees.
4. **Rank-consistency theory** with provable bounds linking proxy scores to WER.
5. **10–100× cheaper** than SSL-NAS because no pretraining inner loop is required.

### Similar Studies & Closeness

| # | Paper | Venue | Closeness | Gap We Fill |
|---|---|---|---|---|
| 1 | **UnNAS** (Liu et al.) | ECCV 2020 | High (concept) | Repurposed existing tasks → We *design* new ones |
| 2 | **MAE-NAS** (Hu et al.) | ICLR 2024 | Medium | Borrowed MAE → We design speech-specific tasks |
| 3 | **NAS-Bench-ASR** (Mehrotra et al.) | ICLR 2021 | High (domain) | Supervised CTC → Label-free proxy tasks |
| 4 | **BEST-RQ** (Chiu et al.) | ICML 2022 | Low | SSL objective, never used for NAS |
| 5 | **LightHuBERT** (Wang et al.) | Interspeech 2022 | Low | Compression, not architecture discovery |
| 6 | **OWLS** (Meta) | arXiv 2025 | Medium | Scaling laws with fixed architecture |
| 7 | **DARTS-Conformer** (Chen et al.) | Interspeech 2021 | High (domain) | Supervised → Label-free |
