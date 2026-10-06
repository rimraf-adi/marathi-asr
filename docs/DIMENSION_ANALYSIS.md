# Structural and Dimensional Analysis Report: Multi-Dialect MoE Conformer ASR

**Document Version**: 1.0.0  
**Target Architecture**: 12-Layer Streaming Conformer with Dynamic-Chunk Causal Attention, Shared-Trunk 3-Expert Mixture-of-Experts (MoE), and Multi-Exit CTC Heads  
**Target Language & Dialects**: Marathi (D3 Standard Marathi) & Regional Dialects (D1 Malvani / Konkan, D2 Ahirani / Khandesh, D4 Varhadi / Vidarbha)  
**Hardware Profile**: NVIDIA RTX A5000 (24 GB VRAM) / Ampere Architecture  

---

## Table of Contents

1. [Executive Summary & High-Level Specifications](#1-executive-summary--high-level-specifications)
2. [End-to-End Tensor Dimensional Trajectory](#2-end-to-end-tensor-dimensional-trajectory)
3. [Deep Module-by-Module Dimensional Breakdown](#3-deep-module-by-module-dimensional-breakdown)
   - [3.1 Log-Mel Front-End Feature Extraction](#31-log-mel-front-end-feature-extraction)
   - [3.2 4x Temporal Subsampling (`Conv2dSubsampling4`)](#32-4x-temporal-subsampling-conv2dsubsampling4)
   - [3.3 Dense Conformer Blocks (Layers 1 – 4)](#33-dense-conformer-blocks-layers-1--4)
   - [3.4 Sparse MoE Conformer Blocks (Layers 5 – 12)](#34-sparse-moe-conformer-blocks-layers-5--12)
   - [3.5 Multi-Exit Taps & CTC Heads](#35-multi-exit-taps--ctc-heads)
   - [3.6 Auxiliary SSL Reconstruction Head](#36-auxiliary-ssl-reconstruction-head)
4. [Parameter Capacity vs. Inference Sparsity](#4-parameter-capacity-vs-inference-sparsity)
5. [Theoretical Computational Complexity (FLOPs & MACs)](#5-theoretical-computational-complexity-flops--macs)
6. [Temporal Resolution, Receptive Field, & Dynamic Chunking](#6-temporal-resolution-receptive-field--dynamic-chunking)
7. [Memory Footprint & Hardware Utilization](#7-memory-footprint--hardware-utilization)
8. [Architectural Summary Reference Card](#8-architectural-summary-reference-card)

---

## 1. Executive Summary & High-Level Specifications

The model is a unified streaming acoustic speech recognition architecture designed specifically to address dialectal variation in Marathi. It combines:
- **4x Convolutional Subsampling**: Condenses 100 Hz spectrogram frames into 25 Hz acoustic representations (40 ms per frame).
- **Macaron-style Conformer Blocks**: Interleaves half-step feed-forward networks, multi-head self-attention with Rotary Position Embeddings (RoPE), and depthwise 1D convolutions.
- **Hierarchical Mixture-of-Experts (MoE)**: In blocks 5–12, the second feed-forward network is upcycled into an always-active Combining Shared Trunk (representing standard Marathi D3) plus 3 dedicated regional dialect experts (D1 Malvani, D2 Ahirani, D4 Varhadi) gated by a frame-level router.
- **Dynamic-Chunk Masking & Multi-Exit CTC Taps**: Enables unified operation across streaming latency regimes ($C \in \{1, 4, 8, 16, -1\}$ frames) and early termination at exit depths 4, 8, or 12.

```
       Raw Audio (16 kHz, 1D Waveform)
                     │
                     ▼
       LogMelSpectrogramFrontEnd (80 filterbanks, 100 fps)
                     │
                     ▼
       Conv2dSubsampling4 (4x downsampling -> 25 fps, 40 ms/frame)
                     │
                     ▼
       PositionalEncoding (Sinusoidal, d_model = 256)
                     │
  ┌──────────────────┴────────────────────────────────────────┐
  │ Layers 1 – 4: Dense Conformer Blocks                      │
  │  • FFN 1 (d_ff = 1024, half-step)                         │
  │  • MHSA (4 heads, d_k = 64, RoPE)                         │
  │  • Conformer Conv Module (kernel = 31, depthwise)         │
  │  • FFN 2 (d_ff = 1024, half-step)                         │
  │  • Final LayerNorm (d_model = 256)                        │
  └──────────────────┬────────────────────────────────────────┘
                     │ ────► [Exit Tap 1 @ Layer 4] (Early Exit)
  ┌──────────────────┴────────────────────────────────────────┐
  │ Layers 5 – 8: Sparse MoE Conformer Blocks                 │
  │  • FFN 1 (d_ff = 1024, half-step)                         │
  │  • MHSA (4 heads, d_k = 64, RoPE)                         │
  │  • Conformer Conv Module (kernel = 31, depthwise)         │
  │  • SparseMoELayer:                                        │
  │     ├── Combining Shared FFN Trunk (D3 Standard Marathi)  │
  │     ├── 3 Dialect Experts (D1 Malvani, D2 Ahirani, D4)    │
  │     └── Lightweight Router (Top-1 Soft / Hard Gating)     │
  │  • Final LayerNorm (d_model = 256)                        │
  └──────────────────┬────────────────────────────────────────┘
                     │ ────► [Exit Tap 2 @ Layer 8] (Intermediate Exit)
  ┌──────────────────┴────────────────────────────────────────┐
  │ Layers 9 – 12: Sparse MoE Conformer Blocks                │
  │  • (Identical structure to Layers 5 – 8)                 │
  └──────────────────┬────────────────────────────────────────┘
                     │ ────► [Exit Tap 3 @ Layer 12] (Final Exit)
                     ▼
           CTCHead (Linear 256 -> 105 vocabulary logits)
```

### Global Hyperparameters

| Hyperparameter | Symbol / Definition | Value | Design Rationale |
| :--- | :--- | :--- | :--- |
| **Audio Sample Rate** | $f_s$ | $16{,}000\text{ Hz}$ | Standard speech bandwidth (Nyquist = 8 kHz) |
| **STFT Window Length** | $N_{win}$ | $400\text{ samples}$ ($25\text{ ms}$) | Balances spectral resolution and stationarity |
| **STFT Hop Size** | $N_{hop}$ | $160\text{ samples}$ ($10\text{ ms}$) | Generates 100 spectrogram frames per second |
| **Mel Filterbank Channels** | $F$ | $80$ channels | Log-mel spectrogram representation |
| **Subsampling Downsampling** | $S_{sub}$ | $4\times$ | Reduces sequence length to 25 fps ($40\text{ ms}$ hop) |
| **Encoder Latent Dimension** | $d_{model}$ | $256$ | Hidden dimension across all Conformer blocks |
| **Attention Heads** | $n_{heads}$ | $4$ | Dimension per head $d_k = d_v = 256 / 4 = 64$ |
| **Feed-Forward Expansion** | $r_{ff}$ | $4\times$ ($d_{ff} = 1024$) | Standard Conformer expansion ratio |
| **Conv Kernel Size** | $K_{conv}$ | $31$ | Temporal context span of $1.2\text{ s}$ per block |
| **Total Encoder Blocks** | $L$ | $12$ | 4 Dense Blocks + 8 Sparse MoE Blocks |
| **MoE Layer Placement** | Indices | Layers 5 to 12 (`[4..11]`) | Deep dialect specialization |
| **Number of Experts** | $N_{experts}$ | $3$ Regional Experts | D1 (Malvani), D2 (Ahirani), D4 (Varhadi) |
| **Shared Trunk FFNs** | $N_{trunk}$ | $1$ Combining Trunk | Always-active Standard Marathi (D3) baseline |
| **CTC Vocabulary Size** | $V$ | $105$ tokens | Character-level Devanagari script + CTC blanks |
| **Multi-Exit Taps** | Taps | Layers $[4, 8, 12]$ | Dynamic compute/accuracy scaling |

---

## 2. End-to-End Tensor Dimensional Trajectory

The table below illustrates the exact shape of tensors flowing through each stage of the network for batch size $B$ over three representative input durations:
- **$1.0\text{ s}$**: $16{,}000$ audio samples
- **$5.0\text{ s}$**: $80{,}000$ audio samples
- **$10.0\text{ s}$**: $160{,}000$ audio samples

| Pipeline Operation | Output Tensor Shape | Shape ($1.0\text{ s}$) | Shape ($5.0\text{ s}$) | Shape ($10.0\text{ s}$) | Time Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Input Audio** | `(B, S)` | `(B, 16000)` | `(B, 80000)` | `(B, 160000)` | $0.0625\text{ ms/sample}$ |
| **Log-Mel Transform** | `(B, T_spec, 80)` | `(B, 101, 80)` | `(B, 501, 80)` | `(B, 1001, 80)` | $10.0\text{ ms/frame}$ (100 fps) |
| **Subsampling Conv2D-1** | `(B, 256, T_spec/2, 40)` | `(B, 256, 51, 40)` | `(B, 256, 251, 40)` | `(B, 256, 501, 40)` | $20.0\text{ ms/frame}$ |
| **Subsampling Conv2D-2** | `(B, 256, T_sub, 20)` | `(B, 256, 26, 20)` | `(B, 256, 126, 20)` | `(B, 256, 251, 20)` | $40.0\text{ ms/frame}$ |
| **Subsampling Flatten** | `(B, T_sub, 256 * 20)` | `(B, 26, 5120)` | `(B, 126, 5120)` | `(B, 251, 5120)` | $40.0\text{ ms/frame}$ |
| **Subsampling Linear Proj** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ (25 fps) |
| **Positional Encoding** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ |
| **Dense Blocks 1 – 4** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ |
| **Exit Tap 1 (Layer 4)** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ |
| **MoE Blocks 5 – 8** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ |
| **Exit Tap 2 (Layer 8)** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ |
| **MoE Blocks 9 – 12** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ |
| **Exit Tap 3 (Layer 12)** | `(B, T_sub, 256)` | `(B, 26, 256)` | `(B, 126, 256)` | `(B, 251, 256)` | $40.0\text{ ms/frame}$ |
| **CTC Projection Logits** | `(B, T_sub, 105)` | `(B, 26, 105)` | `(B, 126, 105)` | `(B, 251, 105)` | $40.0\text{ ms/frame}$ |

### Exact Temporal Frame Mapping Formula
The relationship between input spectrogram time frames $T_{spec}$ and encoder subsampled frames $T_{sub}$ is governed by the two strided 2D convolutions:

$$T_{conv1} = \left\lfloor \frac{T_{spec} + 2 \cdot 1 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{T_{spec} - 1}{2} \right\rfloor + 1$$

$$T_{sub} = \left\lfloor \frac{T_{conv1} + 2 \cdot 1 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{T_{conv1} - 1}{2} \right\rfloor + 1 \approx \left\lceil \frac{T_{spec}}{4} \right\rceil$$

---

## 3. Deep Module-by-Module Dimensional Breakdown

### 3.1 Log-Mel Front-End Feature Extraction
Implemented in `model/asr_model.py:LogMelSpectrogramFrontEnd`.
- Operates on raw audio waveforms $x \in \mathbb{R}^{B \times S}$.
- STFT Transform: $N_{fft} = 400$, window length = $400$ ($25\text{ ms}$), hop length = $160$ ($10\text{ ms}$).
- Mel scale: $80$ triangular filters spanning $0\text{ Hz}$ to $8{,}000\text{ Hz}$.
- Numerical stabilization: $\log(\max(\text{mel\_power}, 10^{-5}))$.
- Parameter count: **$0$ learnable parameters** (deterministic STFT + Mel filter matrix).

---

### 3.2 4x Temporal Subsampling (`Conv2dSubsampling4`)
Implemented in `model/conformer.py:Conv2dSubsampling4`.
Reduces the frame rate from 100 fps to 25 fps via two strided 2D convolutional layers:

1. **Input Reshape**: Unsqueezes channel dimension: $(B, T_{spec}, 80) \to (B, 1, T_{spec}, 80)$.
2. **Conv2D Stage 1**:
   - `nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1)`
   - Kernel Weight: $[256, 1, 3, 3]$ ($2{,}304$ params)
   - Bias: $[256]$ ($256$ params)
   - Output Tensor: $(B, 256, T_{spec}/2, 40)$
   - Non-linearity: `nn.SiLU()` (Swish)
3. **Conv2D Stage 2**:
   - `nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)`
   - Kernel Weight: $[256, 256, 3, 3]$ ($589{,}824$ params)
   - Bias: $[256]$ ($256$ params)
   - Output Tensor: $(B, 256, T_{sub}, 20)$
   - Non-linearity: `nn.SiLU()`
4. **Channel-Frequency Flattening**:
   - Transposes and flattens: $(B, 256, T_{sub}, 20) \to (B, T_{sub}, 256 \times 20) = (B, T_{sub}, 5120)$.
5. **Linear Output Projection**:
   - `nn.Linear(in_features=5120, out_features=256)`
   - Weight: $[256, 5120]$ ($1{,}310{,}720$ params)
   - Bias: $[256]$ ($256$ params)
   - Output Tensor: $(B, T_{sub}, 256)$
- **Submodule Parameter Total**: **$1{,}903{,}616$ parameters** (5.80% of model capacity).

---

### 3.3 Dense Conformer Blocks (Layers 1 – 4)
Implemented in `model/conformer.py:ConformerBlock`.
Each of the first 4 blocks uses dense modules in a Macaron sandwich configuration:

$$\begin{aligned}
\tilde{x}_1 &= x + \frac{1}{2} \operatorname{FFN}_1(x) \\
\tilde{x}_2 &= \tilde{x}_1 + \operatorname{MHSA}(\tilde{x}_1) \\
\tilde{x}_3 &= \tilde{x}_2 + \operatorname{Conv}(\tilde{x}_2) \\
\tilde{x}_4 &= \tilde{x}_3 + \frac{1}{2} \operatorname{FFN}_2(\tilde{x}_3) \\
y &= \operatorname{LayerNorm}(\tilde{x}_4)
\end{aligned}$$

#### Detailed Component Dimensions:

| Sub-Module | Mathematical Operation | Parameter Matrix Shapes | Parameters |
| :--- | :--- | :--- | :--- |
| **FFN 1** | $\text{LN}(x) \to W_1 x + b_1 \to \text{SiLU} \to W_2 h + b_2$ | $\text{LN}: [256], [256]$<br>$W_1: [1024, 256], b_1: [1024]$<br>$W_2: [256, 1024], b_2: [256]$ | $526{,}080$ |
| **MHSA** | $\text{LN}(x) \to Q, K, V \to \text{RoPE}(Q, K) \to \text{Attn}(Q, K) V \to W_o$ | $\text{LN}: [256], [256]$<br>$W_q, W_k, W_v, W_o: 4 \times [256, 256]$<br>$b_q, b_k, b_v, b_o: 4 \times [256]$ | $263{,}168$ |
| **Conv Module** | $\text{LN} \to \text{PW1} \to \text{GLU} \to \text{DW}(K=31) \to \text{GN} \to \text{SiLU} \to \text{PW2}$ | $\text{LN}: [256], [256]$<br>$\text{PW1}: [512, 256, 1], [512]$<br>$\text{DW}: [256, 1, 31], [256]$<br>$\text{GN}: [256], [256]$<br>$\text{PW2}: [256, 256, 1], [256]$ | $206{,}592$ |
| **FFN 2** | Identical structure to FFN 1 | Same as FFN 1 | $526{,}080$ |
| **Final LN** | $\text{LayerNorm}(d_{model} = 256)$ | $\gamma: [256], \beta: [256]$ | $512$ |
| **Total per Dense Block** | **Full Conformer Block (Layers 1–4)** | **Sum of all 5 submodules** | **$1{,}522{,}944$** |

*Total for 4 Dense Blocks (Layers 1–4): $4 \times 1{,}522{,}944 = \mathbf{6{,}091{,}776\text{ parameters}}$.*

---

### 3.4 Sparse MoE Conformer Blocks (Layers 5 – 12)
Implemented in `moe/moe_layer.py:SparseMoELayer`.
In layers 5 through 12, `FFN 2` is replaced with `SparseMoELayer`, maintaining identical dimensions for `FFN 1`, `MHSA`, `Conv Module`, and `Final LayerNorm`.

#### SparseMoELayer Architecture:
1. **Combining Shared Trunk FFN** (Standard Marathi D3 Anchor):
   - Always evaluated for every frame $x \in \mathbb{R}^{B \times T \times 256}$.
   - Structure: `FeedForwardModule(d_model=256, expansion=4)`.
   - Dimensions: Linear $[1024, 256] \to \text{SiLU} \to \text{Linear}[256, 1024]$.
   - Parameters: **$526{,}080$ parameters**.
2. **Dedicated Dialect Experts** (3 Independent Experts):
   - **Expert 0**: D1 (Malvani / Konkan) $\to 526{,}080$ parameters
   - **Expert 1**: D2 (Ahirani / Khandesh) $\to 526{,}080$ parameters
   - **Expert 2**: D4 (Varhadi / Vidarbha) $\to 526{,}080$ parameters
   - Each expert is structurally identical to the combining trunk.
   - Combined Expert Capacity: $3 \times 526{,}080 = \mathbf{1{,}578{,}240\text{ parameters}}$.
3. **Lightweight Gating Router**:
   - `nn.Linear(in_features=256, out_features=3)`
   - Weight: $[3, 256]$ ($768$ params)
   - Bias: $[3]$ ($3$ params)
   - Parameters: **$771$ parameters**.
4. **Mathematical Routing & Additive Delta Fusion**:
   - Router Logits: $z = x W_{router}^T + b_{router} \in \mathbb{R}^{B \times T \times 3}$.
   - Gating Probabilities: $P = \operatorname{softmax}(z, \dim=-1)$.
   - Top-1 Expert Selection: $k^* = \operatorname{argmax}(P, \dim=-1)$, weight $g = \max(P, \dim=-1)$.
   - Additive Innovation Delta:
     $$y_t = \operatorname{FFN}_{combining}(x_t) + g_t \cdot \left(\operatorname{Expert}_{k^*}(x_t) - x_t\right)$$
   *(Subtracting $x_t$ isolates the non-linear adaptation delta $\Delta_k$, preserving the baseline trunk).*

#### MoE Parameter Totals:
- **`SparseMoELayer` Total**: $526{,}080 + 1{,}578{,}240 + 771 = \mathbf{2{,}105{,}091\text{ parameters}}$.
- **Single MoE Conformer Block Total**:
  $$\text{FFN1 } (526{,}080) + \text{MHSA } (263{,}168) + \text{Conv } (206{,}592) + \text{MoE } (2{,}105{,}091) + \text{LN } (512) = \mathbf{3{,}101{,}955\text{ parameters}}$$
- **Total for 8 MoE Blocks (Layers 5–12)**: $8 \times 3{,}101{,}955 = \mathbf{24{,}815{,}640\text{ parameters}}$.

---

### 3.5 Multi-Exit Taps & CTC Heads
Implemented in `model/heads.py:CTCHead`.
- **Tapped Layers**: Layers 4, 8, and 12.
- **CTC Projection Head Structure**:
  - `nn.LayerNorm(normalized_shape=256)`: $\gamma, \beta \in [256]$ ($512$ params)
  - `nn.Dropout(p=0.1)`
  - `nn.Linear(in_features=256, out_features=105)`:
    - Weight: $[105, 256]$ ($26{,}880$ params)
    - Bias: $[105]$ ($105$ params)
- **Output Activation**: `torch.log_softmax(logits, dim=-1)` $\implies (B, T_{sub}, 105)$.
- **Parameters per CTC Head**: **$27{,}497$ parameters**.

---

### 3.6 Auxiliary SSL Reconstruction Head
Implemented in `model/heads.py:MaskedReconstructionHead`.
Used during Stage 1 Self-Supervised Pretraining to invert the 4x subsampling and reconstruct the masked 80-channel log-mel spectrogram frames:
- `Linear 1` ($256 \to 512$): Weight $[512, 256]$, Bias $[512]$ ($131{,}584$ params)
- `SiLU()`
- `LayerNorm(512)`: $1{,}024$ params
- `Linear 2` ($512 \to 4 \times 80 = 320$): Weight $[320, 512]$, Bias $[320]$ ($164{,}160$ params)
- Reshape: $(B, T_{sub}, 320) \to (B, T_{sub} \times 4, 80) \approx (B, T_{spec}, 80)$.
- **Reconstruction Head Parameters**: **$296{,}768$ parameters**.

---

## 4. Parameter Capacity vs. Inference Sparsity

### 4.1 Master Parameter Accounting Table

| Structural Component | Layer Scope | Modules Contained | Total Parameters | Trainable Parameters | Proportion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Log-Mel Front-End** | Input | STFT / Torchaudio Transform | $0$ | $0$ | $0.00\%$ |
| **Conv2dSubsampling4** | Input Stage | 2x Conv2d + SiLU + Linear | $1{,}903{,}616$ | $1{,}903{,}616$ | $5.80\%$ |
| **Positional Encoding** | Embedding | Sinusoidal Buffer | $0$ | $0$ | $0.00\%$ |
| **Dense Conformer Blocks** | Layers 1 – 4 | 4x (FFN1 + MHSA + Conv + FFN2 + LN) | $6{,}091{,}776$ | $6{,}091{,}776$ | $18.55\%$ |
| **MoE Conformer Blocks** | Layers 5 – 12 | 8x (FFN1 + MHSA + Conv + MoE + LN) | $24{,}815{,}640$ | $24{,}815{,}640$ | $75.57\%$ |
| **CTC Projection Head** | Output Head | LayerNorm + Linear(256 $\to$ 105) | $27{,}497$ | $27{,}497$ | $0.08\%$ |
| **TOTAL ASR MODEL** | **All Layers** | **End-to-End Hybrid MoE ASR** | **$32{,}838{,}529$** | **$32{,}838{,}529$** | **$100.00\%$** |

---

### 4.2 Active Inference Parameters per Forward Pass

During inference, only **one regional dialect expert** is gated per token in each MoE block:

$$\begin{aligned}
P_{moe\_active} &= P_{FFN1} + P_{MHSA} + P_{Conv} + P_{FinalLN} + P_{Trunk} + P_{Expert_{k^*}} + P_{Router} \\
&= 526{,}080 + 263{,}168 + 206{,}592 + 512 + 526{,}080 + 526{,}080 + 771 \\
&= \mathbf{2{,}049{,}795\text{ active parameters per MoE block}}
\end{aligned}$$

$$\begin{aligned}
P_{total\_active} &= P_{subsampling} + (4 \times P_{dense\_block}) + (8 \times P_{moe\_active}) + P_{ctc} \\
&= 1{,}903{,}616 + (4 \times 1{,}522{,}944) + (8 \times 2{,}049{,}795) + 27{,}497 \\
&= \mathbf{24{,}421{,}249\text{ active parameters}}
\end{aligned}$$

| Parameter Metric | Count | Ratio / Percentage |
| :--- | :--- | :--- |
| **Total Stored Model Capacity** | **$32{,}838{,}529$** | $100.00\%$ |
| **Active Parameters per Token** | **$24{,}421{,}249$** | $74.37\%$ |
| **Dormant / Unactivated Parameters per Token** | **$8{,}417{,}280$** | $25.63\%$ |
| **Inference Sparsity Ratio** | — | **$25.63\%$ sparse** |
| **Effective Parameter Scaling Multiplier** | — | **$1.345\times$ capacity gain over compute** |

---

## 5. Theoretical Computational Complexity (FLOPs & MACs)

### 5.1 Per-Frame FLOP Decomposition
Expressed in Floating Point Operations (FLOPs) per subsampled frame ($T_{sub}$, 1 step = 40 ms), where 1 Multiply-Accumulate (MAC) = 2 FLOPs:

1. **Feed-Forward Module**:
   - Linear 1 ($256 \to 1024$): $2 \times 256 \times 1024 = 524{,}288$ FLOPs
   - Linear 2 ($1024 \to 256$): $2 \times 1024 \times 256 = 524{,}288$ FLOPs
   - Total per FFN: **$1{,}048{,}576$ FLOPs**
2. **Multi-Head Self-Attention (Projections)**:
   - Projections ($Q, K, V, Out$): $4 \times (2 \times 256 \times 256) = \mathbf{524{,}288\text{ FLOPs}}$
3. **Conformer Convolution Module**:
   - Pointwise 1 ($256 \to 512$): $2 \times 256 \times 512 = 262{,}144$ FLOPs
   - 1D Depthwise Conv ($K=31$): $2 \times 31 \times 256 = 15{,}872$ FLOPs
   - Pointwise 2 ($256 \to 256$): $2 \times 256 \times 256 = 131{,}072$ FLOPs
   - Total Conv Module: **$409{,}088$ FLOPs**
4. **MoE Sub-layer Routing & Forward**:
   - Router ($256 \to 3$): $2 \times 256 \times 3 = 1{,}536$ FLOPs
   - Combining Trunk: $1{,}048{,}576$ FLOPs
   - 1 Active Dialect Expert: $1{,}048{,}576$ FLOPs
   - Total MoE Sub-layer: **$2{,}098{,}688$ FLOPs**

| Block Type | Sub-Modules Involved | FLOPs per Frame |
| :--- | :--- | :--- |
| **Dense Conformer Block** (Layers 1–4) | $\text{FFN1} + \text{MHSA} + \text{Conv} + \text{FFN2}$ | **$3{,}030{,}528\text{ FLOPs}$** |
| **MoE Conformer Block** (Layers 5–12) | $\text{FFN1} + \text{MHSA} + \text{Conv} + \text{MoE}_{active}$ | **$4{,}080{,}640\text{ FLOPs}$** |
| **Encoder Active Total (12 Blocks)** | $4 \times \text{Dense} + 8 \times \text{MoE}$ | **$44{,}767{,}232\text{ FLOPs}$** |

---

### 5.2 Real-Time Compute Throughput (At 25 frames/second)

$$\begin{aligned}
\text{Encoder Active Rate} &= 44{,}767{,}232 \times 25\text{ frames/s} = \mathbf{1.119\text{ GFLOPs/second of audio}} \\
\text{Subsampling Conv Rate} &\approx \mathbf{0.665\text{ GFLOPs/second of audio}} \\
\text{Total Model Compute Rate} &\approx \mathbf{1.784\text{ GFLOPs/second of audio}}
\end{aligned}$$

> **Real-Time Factor (RTF) Analysis**:
> On an NVIDIA RTX A5000 GPU (FP32 peak = $27.8\text{ TFLOPS}$, Tensor Core FP16 peak = $55.6\text{ TFLOPS}$):
> 
> $$\text{Compute Latency for } 1.0\text{ s Audio} = \frac{1.784 \times 10^9\text{ FLOPs}}{27.8 \times 10^{12}\text{ FLOP/s}} \approx 0.064\text{ ms}$$
> 
> The computational Real-Time Factor is **$\text{RTF} < 0.001$**, leaving substantial headroom for multi-channel live streaming and beam-search language model decoding.

---

### 5.3 Multi-Exit Compute Scaling

| Exit Tap | Depth | Executed Blocks | Active FLOPs/Frame | Relative Backbone Compute | Theoretical Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exit 1 (Early)** | Layer 4 | 4 Dense Blocks | $12{,}122{,}112$ | **$27.08\%$** | **$3.69\times$** |
| **Exit 2 (Intermediate)**| Layer 8 | 4 Dense + 4 MoE | $28{,}444{,}672$ | **$63.54\%$** | **$1.57\times$** |
| **Exit 3 (Final)** | Layer 12 | 4 Dense + 8 MoE | $44{,}767{,}232$ | **$100.00\%$** | **$1.00\times$ (Baseline)** |

---

## 6. Temporal Resolution, Receptive Field, & Dynamic Chunking

### 6.1 Receptive Field Propagation
The effective receptive field of the acoustic encoder expands across time through two distinct mechanisms:

1. **2D Strided Convolutions (`Conv2dSubsampling4`)**:
   - Conv1 ($K=3, S=2$): Spans 3 spectrogram frames ($25\text{ ms} + 2 \times 10\text{ ms} = 45\text{ ms}$).
   - Conv2 ($K=3, S=2$): Spans 7 spectrogram frames ($25\text{ ms} + 6 \times 10\text{ ms} = 85\text{ ms}$).
   - Subsampling Stride: 4 spectrogram frames = $40\text{ ms}$ per encoder frame.
2. **Conformer Depthwise Convolutions ($K=31$)**:
   - Each Conformer block applies symmetric padding of 15 frames to the left and 15 frames to the right.
   - At $40\text{ ms}$ per subsampled frame, a single convolution module spans:
     $$(31 - 1) \times 40\text{ ms} = \mathbf{1{,}200\text{ ms}}\text{ (1.20 seconds)}$$
     *(600 ms past context + 40 ms center frame + 600 ms future context)*.
   - Across all 12 layers (neglecting self-attention):
     $$\text{Cumulative Convolution Receptive Field} = 12 \times 1{,}200\text{ ms} = \mathbf{14.4\text{ seconds}}$$

---

### 6.2 Streaming Latency via Dynamic Chunking ($C$)
To enforce causal constraints during live streaming, the attention matrix is masked using chunk sizes $C$ (in subsampled frames):

$$M_{i, j} = \begin{cases} 0 & \text{if } \lfloor j / C \rfloor \le \lfloor i / C \rfloor \\ -\infty & \text{otherwise} \end{cases}$$

| Chunk Config ($C$) | Subsampled Frames ($C$) | Algorithmic Latency (ms) | Training Sampling Weight | Deployment Regime |
| :--- | :--- | :--- | :--- | :--- |
| **$C = 1$** | $1\text{ frame}$ | **$40\text{ ms}$** | $25\%$ | Ultra-low latency interactive dialogue |
| **$C = 4$** | $4\text{ frames}$ | **$160\text{ ms}$** | $25\%$ | Voice assistants / Interactive Voice Response (IVR) |
| **$C = 8$** | $8\text{ frames}$ | **$320\text{ ms}$** | $20\%$ | Live broadcasting / conference captioning |
| **$C = 16$** | $16\text{ frames}$ | **$640\text{ ms}$** | $15\%$ | High-accuracy streaming transcription |
| **$C = -1$** | $\infty$ (Full Utterance) | Unconstrained | $15\%$ | Offline archival batch transcription |

---

## 7. Memory Footprint & Hardware Utilization

### 7.1 Static Parameter Memory

| Representation Format | Bytes / Param | Full Model Storage | Active Parameters Working Set |
| :--- | :--- | :--- | :--- |
| **FP32 (Single Precision)** | $4\text{ bytes}$ | **$125.27\text{ MB}$** | $93.16\text{ MB}$ |
| **FP16 / BF16 (Half Precision)**| $2\text{ bytes}$ | **$62.63\text{ MB}$** | $46.58\text{ MB}$ |
| **INT8 (Quantized)** | $1\text{ byte}$ | **$31.32\text{ MB}$** | $23.29\text{ MB}$ |
| **INT4 (AWQ / GPTQ)** | $0.5\text{ bytes}$ | **$15.66\text{ MB}$** | $11.64\text{ MB}$ |

### 7.2 Activation Memory During Forward Pass
For an input audio segment of $5.0\text{ seconds}$ ($T_{sub} = 126$ frames) at batch size $B=16$ using FP16 precision:
- Spectrogram: $16 \times 501 \times 80 \times 2\text{ bytes} \approx 1.28\text{ MB}$
- Encoder Feature Representations (12 layers $\times 16 \times 126 \times 256 \times 2\text{ bytes}$) $\approx 12.38\text{ MB}$
- Attention Probability Maps (12 layers $\times 16 \times 4 \times 126 \times 126 \times 2\text{ bytes}$) $\approx 24.38\text{ MB}$
- Total Forward Activation Memory: **$\approx 38.04\text{ MB}$** (easily fits on edge devices with $< 2\text{ GB}$ VRAM).

---

## 8. Architectural Summary Reference Card

```
===================================================================================================
                             MOE CONFORMER DIMENSION REFERENCE CARD
===================================================================================================

AUDIO & INPUT FRONT-END:
  Sample Rate:                    16,000 Hz
  STFT Window / Hop:              400 samples (25 ms) / 160 samples (10 ms)
  Filterbank Channels:            80 Log-Mel channels @ 100 fps
  Temporal Subsampling:           4x via 2D Conv (stride 2 x 2) -> 25 fps (40 ms per frame)

ENCODER DIMENSIONS:
  Hidden Dimension (d_model):     256
  Attention Heads (n_heads):      4 (head dimension d_k = d_v = 64)
  Positional Embedding:           Rotary Position Embedding (RoPE) + Sinusoidal
  FFN Expansion (r_ff, d_ff):     4x expansion (d_ff = 1024)
  Conv Module Kernel (K):         31 frames (Depthwise 1D conv, symmetric padding 15)
  Total Encoder Blocks:           12 Blocks (Blocks 1-4 Dense, Blocks 5-12 Sparse MoE)

MIXTURE-OF-EXPERTS (LAYERS 5 - 12):
  Trunk Configuration:            1 Always-Active Combining Shared Trunk (D3 Standard Marathi)
  Dialect Experts:                3 Dedicated Experts (D1 Malvani, D2 Ahirani, D4 Varhadi)
  Router Architecture:            Linear(256 -> 3), Bias(3) [771 parameters]
  Gating Strategy:                Top-1 Soft/Hard Additive Innovation Delta
  Auxiliary Loss:                 Switch Transformer Load Balancing Loss

MULTI-EXIT & VOCABULARY HEADS:
  Exit Taps:                      Layer 4 (Early), Layer 8 (Mid), Layer 12 (Final)
  CTC Vocabulary Size:            105 tokens (Devanagari script + CTC blanks)
  CTC Projection Head:            LayerNorm(256) -> Linear(256 -> 105) [27,497 parameters]

PARAMETERS & INFERENCE COMPUTATION:
  Total Parameters:               32,838,529 (32.84M)
  Active Parameters per Token:    24,421,249 (24.42M) [74.37% Active, 25.63% Sparsity]
  Active FLOPs per Frame:         44,767,232 FLOPs
  Active Inference Compute Rate:  1.784 GFLOPs / second of audio
  Model Storage (FP16):           62.63 MB
  Computational RTF (RTX A5000):  < 0.001
===================================================================================================
```
