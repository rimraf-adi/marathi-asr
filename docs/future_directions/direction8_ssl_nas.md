# Direction 8: Self-Supervised Neural Architecture Search for ASR (SSL-NAS)

## Core Idea
Search ASR encoder architectures using **only unlabeled audio**, then verify that the winners give low WER once fine-tuned with little labeled data. The search objective is an SSL loss (e.g., masked prediction), not CTC/attention.

> [!IMPORTANT]
> Nothing equivalent to UnNAS (vision) exists for speech. Speech SSL (wav2vec 2.0, HuBERT, BEST-RQ) **fixes** the architecture. Almost no one asks *which encoder design suits SSL pretraining*.

---

## 1. Why This is Open

| Domain | Status |
|---|---|
| **Vision NAS** | UnNAS showed label-free search matches supervised search |
| **Speech SSL** | Architecture is always fixed (Conformer/Transformer). No one searches it. |
| **Speech NAS** | Almost always uses labeled data and CTC/attention loss |

The gap: **SSL-objective-driven architecture search for speech encoders does not exist.**

---

## 2. Search Objective Options

### Option A: Masked-Prediction Loss (BEST-RQ Style)
- Random-projection targets — cheap, no tokenizer to train.
- Direct proxy for masked language model quality.

### Option B: Contrastive / InfoNCE Loss
- Ties to mutual-information bounds: $\mathcal{L}_{\text{InfoNCE}} \geq -I(X; Z) + \log K$
- Theoretically grounded but can collapse with poor negatives.

### Option C: Representation-Quality Proxies
- **Effective rank** of learned representations
- **Linear-probe phone accuracy** on unlabeled clustered targets
- **CKA similarity** against a reference model (e.g., frozen HuBERT-Large)

### Option D: Hybrid
- SSL loss for search, then a tiny CTC head on 1h labeled data to calibrate rankings.

---

## 3. Mathematical Formulations

### 3.1 Proxy-to-Downstream Rank Consistency

**The Central Question:** When does the ordering of architectures by SSL loss match their ordering by fine-tuned WER?

Let $\alpha_1, \alpha_2$ be two architectures. Define:
$$\text{SSL-rank:} \quad \mathcal{L}_{\text{SSL}}(\alpha_1) < \mathcal{L}_{\text{SSL}}(\alpha_2)$$
$$\text{WER-rank:} \quad \text{WER}(\alpha_1) < \text{WER}(\alpha_2)$$

**Rank consistency** holds when:
$$P\big(\text{sgn}(\mathcal{L}_{\text{SSL}}(\alpha_1) - \mathcal{L}_{\text{SSL}}(\alpha_2)) = \text{sgn}(\text{WER}(\alpha_1) - \text{WER}(\alpha_2))\big) \geq 1 - \delta$$

**Conditions for consistency** (derive or bound under):
- Linearised / NTK-style model of pretraining
- Mutual-information bound linking masked-prediction loss to downstream risk:
  $$\text{WER}(\alpha) \leq h\big(I(X; Z_\alpha)\big) + \epsilon_{\text{fine-tune}}$$
  where $h$ is a monotonically decreasing function and $I(X; Z_\alpha)$ is the mutual information captured by architecture $\alpha$.

### 3.2 Bilevel Optimisation Structure

**Inner problem:** SSL pretraining of weights $w$ for a given architecture $\alpha$.

**Outer problem:** Architecture selection scored by a downstream proxy.

$$\min_{\alpha \in \mathcal{A}} \quad \mathcal{L}_{\text{SSL}}\big(w^*(\alpha); \alpha\big)$$
$$\text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{SSL}}(w; \alpha)$$

With truncated inner loop (practical):
$$w^*(\alpha) \approx w^{(T)}(\alpha) = \text{SGD}^T\big(w^{(0)}, \nabla_w \mathcal{L}_{\text{SSL}}\big)$$

**Convergence analysis** under non-convex inner loss using truncated unrolled differentiation.

### 3.3 Multi-Objective: SSL Loss + Compute Cost (Pareto NAS)

This strengthens the paper: cost becomes a second objective, so the claim shifts from *"SSL loss ranks architectures"* to **"SSL search finds the accuracy–cost Pareto frontier without labels."**

$$\min_{\alpha \in \mathcal{A}} \quad \Big(\mathcal{L}_{\text{SSL}}(\alpha), \; \text{Cost}(\alpha)\Big)$$

Output is a **Pareto front**, not a single model.

**Cost dimensions to track:**

| Cost Metric | Description |
|---|---|
| **Pretraining cost** | GPU-hours or FLOPs to reach a target SSL loss *(main one)* |
| **Fine-tuning cost** | GPU-hours to reach a target WER on 10h labeled set |
| **Inference cost** | Parameters, FLOPs, real-time factor, peak memory |
| **Search cost** | Total GPU-hours of the NAS itself *(a reviewer will ask)* |
| **Sample efficiency** | SSL loss vs. tokens seen *(distinct axis)* |

**Constrained form** for practical budgets:
$$\min_{\alpha} \quad \mathcal{L}_{\text{SSL}}(\alpha) \quad \text{s.t.} \quad \text{Cost}(\alpha) \leq B$$

**Optimiser choices:** NSGA-II, differentiable supernet with FLOPs penalty (hardware-aware DARTS), or Bayesian optimisation with multi-fidelity.

### 3.4 Compute-Optimal Architecture Theory

Scaling-law work (Chinchilla-style) fixes the architecture and asks how to split budget between parameters and data. **You ask the joint question:**

> *Which architecture family is compute-optimal for SSL pretraining, and does the optimum shift with budget?*

Fit per-architecture scaling laws:
$$\mathcal{L}_{\text{SSL}}(\alpha, C) = A_\alpha \cdot C^{-\beta_\alpha} + L_{\alpha,\infty}$$

where $C$ is compute budget and $A_\alpha, \beta_\alpha, L_{\alpha,\infty}$ are architecture-dependent constants.

**Key finding to aim for:** the compute-optimal architecture changes as budget increases (e.g., Conformer wins at low budget, E-Branchformer at high budget).

### 3.5 Multi-Fidelity Guarantees

Cheap short-run SSL loss as a proxy for long-run loss. Give conditions under which early rankings persist (learning curves don't cross):

$$\text{rank}\big(\mathcal{L}_{\text{SSL}}^{(T_{\text{short}})}(\alpha_i)\big) = \text{rank}\big(\mathcal{L}_{\text{SSL}}^{(T_{\text{full}})}(\alpha_i)\big) \quad \forall i$$

This directly cuts search cost by allowing early stopping of unpromising architectures.

### 3.6 Failure Characterisation

SSL objectives can **collapse** or be **shortcut-solvable** (e.g., easy local-context prediction). A theory of when the objective is *architecture-discriminative* is a real contribution:

- Define discriminativeness: $\text{Var}_\alpha[\mathcal{L}_{\text{SSL}}(\alpha)]$ must be large relative to noise.
- Identify when masked prediction degenerates (e.g., trivial next-frame copy with large conv kernels).

### 3.7 PAC-Bayes Generalisation Bound

Over a distribution of architectures $\rho(\alpha)$, with the SSL loss as the empirical risk:
$$\mathbb{E}_{\alpha \sim \rho}\big[\text{WER}(\alpha)\big] \leq \mathbb{E}_{\alpha \sim \rho}\big[\mathcal{L}_{\text{SSL}}(\alpha)\big] + \sqrt{\frac{D_{\text{KL}}(\rho \| \pi) + \ln(2n/\delta)}{2n}}$$

---

## 4. Search Space

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

## 5. Experimental Plan

1. **Tabular benchmark:** Sample 100–200 architectures, pretrain each briefly with BEST-RQ-style masking. Fine-tune each with CTC on 10h (LibriLight/LibriSpeech).
2. **Correlation analysis:** Measure Spearman/Kendall $\tau$ between SSL loss and WER. Check whether correlations hold across data scale and domain.
3. **Pareto comparison:** Compare Pareto fronts — SSL-searched vs. supervised-searched vs. random vs. hand-designed Conformer.
4. **Budget regimes:** Add a low-budget regime (small models, short pretraining) and a high-budget one. Show whether the winning architecture changes.
5. **Total pipeline cost:** Report search + pretrain + fine-tune, against supervised NAS. The claim *"label-free and cheaper overall"* is very reviewer-friendly.
6. **Out-of-domain + multilingual:** Evaluate on SUPERB tasks for generality.

**Report:** hypervolume and top-k regret against the true front from the benchmark.

---

## 6. Headline Claims

> **(a)** SSL loss is a reliable ranking signal for ASR architectures, with theory saying when.
>
> **(b)** The searched encoders beat baselines at matched compute.
>
> **(c)** The architectures transfer across languages and scales.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| SSL loss correlates weakly with WER at the top of the ranking | Characterise it, fix with calibrated proxy. Still publishable. |
| GPU-hours are noisy and hardware-dependent | Use FLOPs and memory as primary, wall-clock as secondary. State hardware. |
| SSL objective shortcuts (collapse, trivial prediction) | Add discriminativeness analysis; use BEST-RQ (random projections resist collapse) |

---

## Novelty & Literature Context

### What has been done?
- **UnNAS (Liu et al., 2020):** Showed label-free NAS matches supervised NAS in vision using rotation/jigsaw/colorization proxies. *Speech: Nothing equivalent.*
- **AutoSpeech (Ding et al., 2020):** NAS for speech, but uses supervised CTC loss.
- **EfficientTDNN (Ding et al., 2021):** NAS for speaker verification, supervised.
- **BEST-RQ (Chiu et al., 2022):** Random-projection SSL for speech, but fixes the Conformer architecture.
- **Chinchilla (Hoffmann et al., 2022):** Compute-optimal scaling for LLMs. *Speech: No equivalent study.*

### What is NOVEL in our approach?
- **First SSL-objective-driven architecture search for speech encoders.**
- **Per-architecture scaling laws for speech SSL** (Chinchilla-style, but architecture is a variable).
- **Formal rank-consistency theory** linking SSL loss to downstream WER.
- **Multi-objective Pareto NAS** without any labels.

### Similar Studies & Closeness
1. **"UnNAS: Label-Free NAS" (Liu et al., NeurIPS 2020)**
   - *Closeness:* High (concept), Low (domain). Vision only. We bring this to speech.
2. **"BEST-RQ: SSL Pretraining for ASR" (Chiu et al., 2022)**
   - *Closeness:* Medium. Provides the SSL objective we use, but never searches architectures.
3. **"AutoSpeech: NAS for Speech Recognition" (Ding et al., Interspeech 2020)**
   - *Closeness:* High (domain), Low (method). Uses supervised CTC loss. We replace it with SSL.
