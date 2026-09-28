# Direction 9: Robustness-Driven NAS with Label-Free Invariance Objectives

## Core Idea
Search ASR encoder architectures by maximising **representation invariance** under acoustic perturbations, using only unlabeled audio. No pretraining task (no masked prediction), avoiding SSL collapse. The architecture that produces the most stable representations under noise, reverb, accent shift, and codec distortion is selected as the winner.

> [!IMPORTANT]
> This is SSL-adjacent because it uses unlabeled audio and representation consistency, but there is no pretraining task. This avoids masked prediction and its collapse/shortcut issues entirely.

---

## 1. Formal Problem Statement

### 1.1 Minimax-Bilevel Formulation

$$\min_{\alpha \in \mathcal{A}} \; \max_{\delta \in \Delta} \quad R(\alpha, \delta) + \lambda \cdot \text{Cost}(\alpha)$$

$$\text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)$$

where:
$$R(\alpha, \delta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{unlabeled}}} \Big[ d\big( f_{w^*(\alpha)}(x), \; f_{w^*(\alpha)}(T_\delta(x)) \big) \Big]$$

| Symbol | Meaning |
|---|---|
| $\alpha$ | Architecture (from search space $\mathcal{A}$) |
| $w^*(\alpha)$ | Trained weights for architecture $\alpha$ (inner problem) |
| $T_\delta$ | Perturbation operator parameterised by $\delta$ |
| $\Delta$ | Perturbation set (noise, reverb, speed, pitch, codec, SpecAugment, accent) |
| $d(\cdot, \cdot)$ | Representation distance (cosine, CKA, or CTC-posterior KL) |
| $\lambda$ | Cost trade-off coefficient |

This is a **minimax-bilevel problem**, which is more general than plain bilevel NAS.

### 1.2 Perturbation Family $T$

| Perturbation | Parameters | Type |
|---|---|---|
| Additive noise (babble, white, traffic) | SNR ∈ [0, 20] dB | Acoustic |
| Room impulse response (reverb) | RT60 ∈ [0.2, 1.5] s | Acoustic |
| Speed perturbation | Factor ∈ [0.85, 1.15] | Temporal |
| Pitch shift | Semitones ∈ [-3, +3] | Spectral |
| Codec compression | Opus 8–64 kbps, AMR-WB | Channel |
| SpecAugment | Random freq/time masks | Regularisation |
| Accent / dialect shift | Speaker from different dialect | Linguistic |

---

## 2. Label-Free Proxies

All proxies below require **zero labeled data**:

### 2.1 Invariance Score
Embedding drift under perturbations:
$$\text{Inv}(\alpha) = \mathbb{E}_{x, \delta}\Big[1 - \cos\big(f_\alpha(x), f_\alpha(T_\delta(x))\big)\Big]$$

Lower is better. Minimise this over architectures.

### 2.2 Consensus / Self-Distillation Agreement
Agreement between the candidate encoder and an EMA teacher on unlabeled audio:
$$\text{Consensus}(\alpha) = \mathbb{E}_x\Big[\text{KL}\big(p_{\text{EMA}}(\cdot|x) \;\|\; p_\alpha(\cdot|x)\big)\Big]$$

### 2.3 Pseudo-Label Stability
CTC posterior consistency across augmentations (noisy-student style):
$$\text{Stability}(\alpha) = \mathbb{E}_{x, \delta}\Big[\text{KL}\big(\text{CTC}_\alpha(x) \;\|\; \text{CTC}_\alpha(T_\delta(x))\big)\Big]$$

### 2.4 Geometric Proxies
- **Lipschitz constant** $\text{Lip}(f_\alpha)$: smaller → more robust
- **Hessian trace** $\text{tr}(\nabla^2 \mathcal{L})$: sharpness of the loss landscape
- **Effective rank** of representations: $\text{erank}(Z) = \exp\big(-\sum_i \bar{\sigma}_i \log \bar{\sigma}_i\big)$

> [!WARNING]
> Invariance can be trivially satisfied by collapsed or low-information representations (constant function has perfect invariance). Fix by adding a **non-collapse term**: variance regulariser or effective rank lower bound.

**Non-collapse constraint:**
$$\text{erank}\big(f_\alpha(X)\big) \geq r_{\min}$$

---

## 3. Mathematical Novelty

### 3.1 Robustness Certificate by Architecture (Lipschitz Bound)

Bound the worst-case representation drift by a product of per-block Lipschitz constants.

For a $L$-layer encoder $f_\alpha = g_L \circ g_{L-1} \circ \cdots \circ g_1$:
$$\| f_\alpha(x) - f_\alpha(T_\delta(x)) \| \leq \left(\prod_{\ell=1}^{L} \text{Lip}(g_\ell; \alpha_\ell)\right) \cdot \| x - T_\delta(x) \|$$

where $\text{Lip}(g_\ell; \alpha_\ell)$ depends on the architectural choices at layer $\ell$ (attention heads, conv kernel, FFN width).

**This gives a differentiable, architecture-dependent regulariser:**
$$\mathcal{R}_{\text{Lip}}(\alpha) = \sum_{\ell=1}^L \log \text{Lip}(g_\ell; \alpha_\ell)$$

And a **search-space-level theorem:**
> *For architecture $\alpha$ with per-block Lipschitz constants $\{L_\ell\}$, the worst-case representation drift under any perturbation $\delta$ with $\|T_\delta(x) - x\| \leq \epsilon$ is bounded by $\epsilon \prod_\ell L_\ell$.*

### 3.2 Rank Consistency Under Domain Shift

Conditions under which the invariance ordering matches the out-of-domain WER ordering:

$$\text{rank}_{\text{Inv}}(\alpha_i) \approx \text{rank}_{\text{WER-OOD}}(\alpha_i) \quad \text{when:}$$

1. The perturbation family $\Delta$ covers the actual domain shift distribution.
2. The non-collapse constraint ensures representations carry sufficient information.
3. The downstream task (CTC) is Lipschitz-continuous w.r.t. the encoder output.

Under these conditions, generalisation claim becomes **provable rather than just empirical**.

### 3.3 Minimax-Bilevel Convergence

Convergence rates for alternating updates of $\alpha$, $w$, and the adversary $\delta$:

$$\alpha^{(t+1)} = \alpha^{(t)} - \eta_\alpha \nabla_\alpha R(\alpha^{(t)}, \delta^{(t)})$$
$$\delta^{(t+1)} = \Pi_\Delta \big[\delta^{(t)} + \eta_\delta \nabla_\delta R(\alpha^{(t)}, \delta^{(t)})\big]$$
$$w^{(t+1)} = w^{(t)} - \eta_w \nabla_w \mathcal{L}_{\text{train}}(w^{(t)}, \alpha^{(t)})$$

Under Polyak-Łojasiewicz (PL) condition on the inner problem or with truncated-inner-loop assumptions, derive convergence rate:
$$R(\alpha^{(T)}, \delta^{(T)}) - R^* \leq O\big(T^{-1/2}\big)$$

### 3.4 Discretisation Gap Under Robust Objective

Does the robust relaxation reduce DARTS collapse?

**Claim:** Flat, robust optima in the continuous relaxation are less prone to the discretisation gap than sharp, non-robust optima.

**Testable prediction:** The Hessian eigenvalue spectrum of the architecture parameters $\alpha$ is flatter when trained with the robust objective vs. standard DARTS loss.

### 3.5 DRO Generalisation Bound

Over a Wasserstein ball of domains $\mathcal{B}_\rho(P_{\text{train}})$:
$$\sup_{Q \in \mathcal{B}_\rho(P_{\text{train}})} \mathbb{E}_{x \sim Q}\big[\mathcal{L}(f_\alpha(x), y)\big] \leq \mathbb{E}_{x \sim P_{\text{train}}}\big[\mathcal{L}(f_\alpha(x), y)\big] + \rho \cdot \text{Lip}(f_\alpha) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

The architecture $\alpha$ enters through $\text{Lip}(f_\alpha)$, making it a **searchable variable** in the DRO bound.

---

## 4. How This Differs From SSL-NAS (Direction 8)

| Aspect | SSL-NAS (Direction 8) | Robustness-NAS (This Direction) |
|---|---|---|
| **Proxy** | Masked-prediction / contrastive loss | Representation invariance under perturbation |
| **Pretraining required?** | Yes (SSL inner loop) | No (just training with any objective) |
| **Cost** | Expensive (full pretrain per architecture) | Cheaper (no pretrain loop) |
| **Collapse risk** | SSL shortcuts possible | Trivial invariance via collapse (fixable with rank constraint) |
| **Target property** | General representation quality | Specifically out-of-domain robustness |
| **Composability** | Standalone | Can combine with SSL as add-on |

---

## 5. Experimental Plan

1. **Reuse architecture benchmark:** Same 100–200 Conformer-family architectures from Direction 8.
2. **Robustness table:** For each architecture, measure WER on clean, noisy (SNR 0/5/10/15/20 dB), reverbed, accented, and cross-domain test sets.
3. **Proxy correlation:** Correlate each proxy (invariance, consensus, stability, Lipschitz, effective rank) with out-of-domain WER. Report top-k regret.
4. **Search comparison:** Search with the robust objective and compare against:
   - Supervised NAS
   - SSL-NAS (Direction 8)
   - Random search
   - Hand-designed Conformer
   - All at matched compute budget.
5. **Ablate perturbation families:** Show which perturbation types matter most.
6. **Lipschitz certificate validation:** Verify that the theoretical bound predicts the actual robustness ranking.

---

## 6. Headline Claims

> **(a)** Representation invariance under perturbations is a label-free, pretrain-free proxy for out-of-domain ASR quality.
>
> **(b)** The Lipschitz bound provides a provable, architecture-dependent robustness certificate.
>
> **(c)** Robust NAS avoids DARTS collapse and yields flatter, more transferable optima.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Trivial invariance via representation collapse | Enforce effective rank lower bound $\text{erank}(Z) \geq r_{\min}$ |
| Perturbation set doesn't cover real domain shifts | Use diverse, measured perturbations; validate on held-out domains |
| Minimax optimisation instability | Use PGD for inner adversary with warm restarts; smooth $\max$ approximation |
| Lipschitz bound too loose to be useful | Report both bound and empirical drift; tighten via spectral normalisation |

---

## Novelty & Literature Context

### What has been done?
- **Adversarial NAS (Guo et al., 2020):** NAS with adversarial training for image classifiers. *Speech: Nothing equivalent.*
- **RobNet (Guo et al., ECCV 2020):** Searched architectures for adversarial robustness in vision.
- **DRO for ASR:** Minimal work. Some domain-adaptive ASR uses group DRO but not architecture search.
- **Lipschitz-constrained networks:** Studied in vision (LipSDP, orthogonal layers) but not applied to speech architecture search.

### What is NOVEL in our approach?
- **First robustness-driven, label-free architecture search for speech.**
- **Lipschitz robustness certificate parameterised by architectural choices** — a search-space-level theorem, not just weight-level.
- **Minimax-bilevel formulation** that is more general than standard bilevel NAS.
- **Formal connection** between invariance proxy and out-of-domain WER via the DRO bound.

### Similar Studies & Closeness
1. **"RobNet: Robust NAS under Adversarial Attacks" (Guo et al., ECCV 2020)**
   - *Closeness:* High (concept), Low (domain). Vision only, $\ell_p$ adversarial perturbations. We use acoustic perturbations for speech.
2. **"Understanding Robustness of Transformers for Image Classification" (Bhojanapalli et al., ICCV 2021)**
   - *Closeness:* Medium. Analyses Lipschitz properties of vision transformers. We extend this theory to speech encoder search spaces.
3. **"DRO for Domain Generalisation in ASR" (Various)**
   - *Closeness:* Medium. Uses DRO at the training level. We use it at the **architecture search** level.
