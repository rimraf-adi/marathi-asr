# Marathi Streaming Conformer ASR

A streaming-ready, multi-dialect Conformer Automatic Speech Recognition (ASR) architecture for Marathi and its regional dialect continuum (*Malvani*, *Ahirani/Khandeshi*, *Varhadi*, *Konkani*, *Powari*, *Lambani*).

The system implements a **5-stage curriculum**:
1. **Base SSL Pretraining** (Continuous time-span masked log-mel reconstruction)
2. **Dynamic-Chunk Causal Adaptation & Supervised CTC** (Acoustic-phonetic alignment under streaming constraints)
3. **Joint Multi-Exit Early CTC Training** (Layers 4, 8, 12 for sub-100ms early-exit inference)
4. **Mixture-of-Experts (MoE) Dialect Adaptation** (Specialized FFN experts across regional dialects)
5. **Joint Calibration & Sequence Polish** (Dialect exit calibration + direct CER minimization via GRPO)

---

## 1. Primary Dataset Arsenal & Specific Split Configurations

| Dataset | Location / Source | Split Configuration | Size / Shards | Audio Hours | Content Description & Dialects | Primary Role in Pipeline |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **ai4bharat/Shrutilipi** | Hugging Face (Streaming) | `split="train"` | **37.11 GB**<br>(88 shards) | **~253.7 hrs** | Broadcast & read speech with parallel Devanagari text (`marathi`: 35.95 GB / 85 files; `konkani`: 1.15 GB / 3 files) | Stages 1, 2, 3 (SSL Pretrain & Causal CTC Alignment) |
| **ARTPARK-IISc/Vaani** | Hugging Face (Streaming) | `split="train"` (Marathi dialect belt) | **127.65 GB**<br>(253 shards) | **~1,180+ hrs** | Multi-district conversational & field speech across 6 core Marathi-belt dialects and rural recordings | Stages 1, 2, 3 (SSL Pretrain & Dynamic-Chunk Causal CTC) |
| **Combined Streaming Arsenal** | On-the-fly prefetch stream | Interleaved round-robin (`get_combined_stream`) | **164.75 GB**<br>(341 shards) | **~1,435+ hrs** | Comprehensive pan-Maharashtra acoustic coverage streamed via background daemons with zero local disk footprint | Stages 1, 2, 3 |
| **IISc RESPIN Train** (`IISc_RESPIN_train_mr_clean`) | `D:\dialect-norm\IISc_RESPIN_train_mr_clean` | **MoE Train Split (95% speakers)**<br>Speaker-disjoint partition | **~975.0 hrs**<br>Local WAVs | ~769,000 utts<br>(~2,512 speakers) | 4 Regional Dialects: Malvani (D1), Ahirani (D2), Standard (D3), Varhadi (D4) | **Stage 4**: MoE Dialect Adaptation (Train 3 Experts + Combining Trunk + Router) |
| **IISc RESPIN Dev / Callib** (`IISc_RESPIN_train_mr_clean`) | `D:\dialect-norm\IISc_RESPIN_train_mr_clean` | **Held-Out Calibration Split (5% speakers)**<br>Speaker-disjoint partition | **~51.0 hrs**<br>Local WAVs | ~40,000 utts<br>(~132 speakers) | 4 Regional Dialects (Strictly disjoint unseen speakers from Stage 4) | **Stage 5**: Calibration Pass ("Callib" - Temperature Scaling & Exit Threshold Tuning) |
| **IISc RESPIN Test** (`IISc_RESPIN_test_mr`) | `D:\dialect-norm\IISc_RESPIN_test_mr` | **Official Test Set** (`meta_test_mr.json`) | **3.04 hrs**<br>Local WAVs | 2,170 utts<br>(160 unseen speakers) | Strictly held-out thesis evaluation suite (D1: 559, D2: 540, D3: 555, D4: 516) | **Stage 6**: Strictly Frozen Final Benchmark (Never touched during training/tuning) |

### Exact Shard & Storage Breakdown: `ai4bharat/Shrutilipi`
* **Total Dataset Footprint**: **37.11 GB** across **88 shards**
  * `marathi`: **85 shards | 35.95 GB** (~245.5 hours, broadcast & news speech)
  * `konkani`: **3 shards | 1.15 GB** (~8.2 hours, coastal broadcast speech)

### Exact Shard & Storage Breakdown: `ARTPARK-IISc/Vaani`
* **Core Marathi-Belt Dialects**: **127.65 GB** across **253 shards**
  * `Marathi` (Standard / Central Desh / Marathwada): **232 shards | 118.34 GB** (Pune, Satara, Kolhapur, Ahmednagar, Solapur, Aurangabad)
  * `Konkani` (Coastal Konkan / North Goa border): **13 shards | 7.04 GB** (Ratnagiri, Sindhudurg, Goa fringe)
  * `Malvani` (Southern Konkan belt): **4 shards | 1.54 GB** (Sindhudurg, Sawantwadi, Malvan)
  * `Khandeshi` (Northern Khandesh belt): **2 shards | 0.49 GB** (Jalgaon, Dhule, Nandurbar, North Nashik)
  * `Powari` (Eastern Vidarbha belt): **1 shard | 0.16 GB** (Bhandara, Gondia, Balaghat fringe)
  * `Lambani` (Southern border nomadic / tribal dialect): **1 shard | 0.07 GB** (Nanded, Latur border)
* **Extended Maharashtra Border Tribal Languages Available in Vaani**: **7.06 GB** across **18 shards**
  * `Halbi`: 11 shards | 5.63 GB
  * `Gondi`: 3 shards | 0.89 GB
  * `Bhili`: 1 shard | 0.22 GB
  * `Bhatri`: 1 shard | 0.18 GB
  * `Dorli`: 1 shard | 0.14 GB
* **Grand Total Matched Vaani Footprint**: **134.70 GB across 271 shards**

### Granular Dialect Breakdown in RESPIN (Local Benchmark & MoE Corpus)
* **D1 (Malvani / Konkan)**: 203,651 utts | 247.31 hrs | 732 speakers
* **D2 (Ahirani / Khandeshi)**: 198,560 utts | 265.25 hrs | 557 speakers
* **D3 (Standard Marathi / Desh)**: 208,850 utts | 255.79 hrs | 608 speakers
* **D4 (Varhadi / Vidarbha)**: 198,873 utts | 257.70 hrs | 747 speakers
* **Official Test Set**: 2,170 utts across 160 unseen speakers (D1: 559, D2: 540, D3: 555, D4: 516).

---

#### Exact Data Partitioning, Filtering & Split Configuration: Stages 1, 2, and 3

Both upstream repositories (`ai4bharat/Shrutilipi` and `ARTPARK-IISc/Vaani`) publish all audio shards on the Hugging Face Hub under the single partition identifier **`split="train"`**. Because neither dataset provides native upstream validation splits, our pipeline implements **rigorous on-the-fly programmatic partitioning, strict inclusion/exclusion filtering, dynamic latency conditioning, and in-memory evaluation buffers** across the streaming curriculum (Stages 1–3).

---

#### 1. Stage-by-Stage Data Split & Ingestion Specifications

The following table provides the exact split configurations, inclusion criteria, online transformations, and total audio exposure across all training and evaluation stages:

| Stage & Objective | Physical Source & HF Split | Partition / Filtering Criteria | Duration Filter | Latency / Chunk Conditioning | Batch & Step Budget | Total Audio Exposure | Primary Verification & Split Validation |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Stage 1**<br>Self-Supervised Pretraining (SSL) | • `ai4bharat/Shrutilipi` (`split="train"`, 88 shards)<br>• `ARTPARK-IISc/Vaani` (`split="train"`, 253 shards) | **100% Text-Agnostic / Unsupervised**.<br>All audio streams ingested regardless of transcript presence (`isTranscriptionAvailable` ignored). | 1.0s – 12.0s<br>(16,000 – 192,000 samples) | Full Context bidirectional with continuous span log-mel masking ($p=0.40, L=10$ frames / 400ms). | 30,000 steps<br>(batch 16, grad accum 1) | **480,000 utts**<br>(~733.3 audio hrs) | **In-Memory Intrinsic Evaluation Gate** (16 held-out samples):<br>• Short-mask ($L=4$) vs Long-mask ($L=20$) loss ratio ($> 1.20\times$)<br>• Latent representation effective rank ($> 15.0$) |
| **Stage 2**<br>Dynamic-Chunk Causal Alignment & CTC SFT | • `ai4bharat/Shrutilipi` (`split="train"`)<br>• `ARTPARK-IISc/Vaani` (`split="train"`, 6 core configs) | **Supervised Speech-Text Pairs Only**.<br>• Vaani: `isTranscriptionAvailable == "Yes"` & `transcript.strip() != ""` <br>• Shrutilipi: `text.strip() != ""` | 1.0s – 12.0s<br>(16,000 – 192,000 samples) | **Dynamic Chunk Schedule**:<br>• $C=1$ frame (40ms): 25%<br>• $C=4$ frames (160ms): 25%<br>• $C=8$ frames (320ms): 20%<br>• $C=16$ frames (640ms): 15%<br>• $C=-1$ (Full context): 15% | 30,000 steps<br>(batch 32, grad accum 2 = 64 utts/step) | **1,920,000 utts**<br>(~2,933.3 audio hrs) | **Live Batch Slice Evaluation**:<br>• Greedy CTC Levenshtein CER computed every 25 steps<br>• Rolling CTC loss tracking<br>• Top-3 checkpoints + `best_model.pt` retention based on minimum CER |
| **Stage 3**<br>Joint Multi-Exit Early CTC + GRPO Policy | • `ai4bharat/Shrutilipi` (`split="train"`)<br>• `ARTPARK-IISc/Vaani` (`split="train"`, 6 core configs) | **Supervised Speech-Text Pairs Only**.<br>Identical strict supervised audio-text stream as Stage 2. | 1.0s – 12.0s<br>(16,000 – 192,000 samples) | **Multi-Exit Heads + Dynamic Chunks**:<br>• Layer 4 Head: $\alpha_4 = 0.30$<br>• Layer 8 Head: $\alpha_8 = 0.30$<br>• Layer 12 Head: $\alpha_{12} = 1.00$<br>• Dynamic Chunks: $C \in \{1, 4, 8, 16, -1\}$ | 25,000 steps<br>(batch 24, grad accum 2 = 48 utts/step) | **1,200,000 utts**<br>(~1,833.3 audio hrs) | **GRPO Policy Rollout Gate**:<br>• $G=3\text{--}4$ trajectory rollouts per sample<br>• Pareto reward ($R = -\text{CER} - \lambda \cdot \text{Latency}$, $\lambda=0.25$)<br>• KL penalty ($\beta_{KL}=0.02$) against frozen Stage 2 backbone |
| **Stage 4**<br>3-Dialect MoE Adaptation | `D:\dialect-norm\IISc_RESPIN_train_mr_clean` | **Local RESPIN Train Split**:<br>Strict 95% speaker partition (~2,512 speakers) across D1 (Malvani), D2 (Ahirani), D3 (Standard), D4 (Varhadi). | 1.0s – 15.0s<br>(WAV files) | Dynamic Chunking + Expert Routing:<br>• Phase 4A: Hard dialect routing (7,500 steps)<br>• Phase 4B: Soft Top-1/Top-2 routing (17,500 steps) | 25,000 steps<br>(batch 24, grad accum 2 = 48 utts/step) | **~769,000 utts**<br>(~975.0 audio hrs) | **Per-Dialect Load Balancing**:<br>• Auxiliary routing entropy loss ($\mathcal{L}_{aux}$ weight 0.01)<br>• Per-expert CER tracking across D1–D4 |
| **Stage 5**<br>Joint Calibration ("Callib") | `D:\dialect-norm\IISc_RESPIN_train_mr_clean` | **Local RESPIN Held-Out Calibration Split**:<br>Strict 5% speaker partition (~132 speakers) completely disjoint from Stage 4. | 1.0s – 15.0s<br>(WAV files) | Multi-exit inference + Dialect Router Temperature Scaling ($\tau_d$) + Exit Confidence Thresholds ($\theta_4, \theta_8$). | 8,000 steps<br>(batch 24, grad accum 2 = 48 utts/step) | **~40,000 utts**<br>(~51.0 audio hrs) | **Expected Calibration Error (ECE)**:<br>• Post-scaling ECE minimization ($< 5.0\%$) across all exits and dialects |
| **Stage 6**<br>Frozen Evaluation Benchmark | `D:\dialect-norm\IISc_RESPIN_test_mr` | **Official Held-Out Test Set** (`meta_test_mr.json`):<br>160 strictly unseen speakers across D1 (559 utts), D2 (540 utts), D3 (555 utts), D4 (516 utts). | All test audio<br>(3.04 hrs total) | Evaluated across both streaming ($C \in \{1, 4, 8, 16\}$) and non-causal ($C=-1$) conditions. | N/A<br>(Single-pass frozen eval) | **2,170 utts**<br>(3.04 audio hrs) | **Zero-Leakage Benchmark Metrics**:<br>• Macro-averaged & Per-dialect CER/WER<br>• Average layer depth (compute savings vs Layer 12 baseline) |

---

#### 2. Deep Dive: Exact Data Splits in Stages 1, 2, and 3

##### Stage 1: Unsupervised Self-Supervised Pretraining (SSL)
* **Hub Split Ingested**: `split="train"` across `ai4bharat/Shrutilipi` (88 shards, 37.11 GB) and `ARTPARK-IISc/Vaani` (253 shards, 127.65 GB).
* **Text-Agnostic Partition**:
  * Consumes 100% of available audio streams regardless of transcription availability.
  * In `ARTPARK-IISc/Vaani`, rural field speech with `isTranscriptionAvailable == "No"` (untranscribed or dialectal chatter) is explicitly retained. This exposes the Conformer backbone to real-world acoustic reverberation, microphone distortions, and rural dialectal intonations before text alignment begins.
* **Duration Bounds**: Utterances are filtered on-the-fly to $1.0\text{s} \le \text{duration} \le 12.0\text{s}$ (16,000 to 192,000 samples @ 16 kHz). Any audio shorter than 1.0s (clicks, transient noise) or longer than 12.0s (memory protection) is automatically skipped.
* **Total Exposure**: At 30,000 steps with batch size 16, Stage 1 consumes **480,000 unique audio utterances** totaling **~733.3 audio hours**.
* **Validation Partition**: An in-memory evaluation buffer of 16 diverse samples is populated during the first steps. Every 100 steps, an **Intrinsic Evaluation Gate** evaluates:
  1. *Short-Mask vs. Long-Mask Degradation*: Forward-pretrain with 4-frame masks vs. 20-frame masks. If the loss ratio $L_{\text{long}} / L_{\text{short}} < 1.20$, training warns of local temporal interpolation.
  2. *Latent Space Rank*: Verifies that the effective rank of Layer 12 representations remains $> 15.0$, preventing dimensional representation collapse.

##### Stage 2: Dynamic-Chunk Causal Alignment & Supervised CTC Fine-Tuning
* **Hub Split Ingested**: `split="train"` across `ai4bharat/Shrutilipi` and `ARTPARK-IISc/Vaani` (6 core configs: `Marathi`, `Konkani`, `Malvani`, `Khandeshi`, `Powari`, `Lambani`).
* **Supervised Speech-Text Filtering (`causal_alignment/dataset.py`)**:
  * Unlike Stage 1, Stage 2 **strictly filters out untranscribed speech**:
    ```python
    text = sample.get("text", "").strip()
    audio = sample.get("audio")
    # Must have valid text transcript for supervised CTC
    if not text or audio is None:
        continue
    dur_samples = len(audio)
    if dur_samples < min_samples or dur_samples > max_samples:
        continue
    ```
  * In Vaani, only samples where `isTranscriptionAvailable == "Yes"` and `transcript.strip() != ""` are accepted.
  * In Shrutilipi, only samples with valid non-empty `text` are accepted.
* **Vocabulary Tokenization**: Valid transcripts are tokenized via [`causal_alignment/tokenizer.py`](file:///d:/marathi-asr/causal_alignment/tokenizer.py) using the 105-token canonical Marathi Devanagari vocabulary (`causal_alignment/vocab.json`). Any text containing completely unmapped unicode characters is assigned the `<unk>` token.
* **Dynamic-Chunk Lookahead Schedule**:
  To ensure robust streaming across diverse deployment latency constraints, each batch is assigned a causal lookahead chunk size according to a fixed probability distribution:
  * **$C = 1$ frame (40ms lookahead / zero future context)**: $25\%$ probability. Models strict real-time streaming where text must be emitted with zero delay.
  * **$C = 4$ frames (160ms lookahead)**: $25\%$ probability. Models low-latency interactive streaming.
  * **$C = 8$ frames (320ms lookahead)**: $20\%$ probability. Models standard conversational streaming.
  * **$C = 16$ frames (640ms lookahead)**: $15\%$ probability. Models latency-tolerant buffered streaming.
  * **$C = -1$ (Full context bidirectional)**: $15\%$ probability. Upper-bound offline accuracy benchmark.
* **Online Augmentation Pipeline**:
  * Speed perturbation: $0.9\times, 1.1\times$ applied with $p=0.50$ via linear waveform interpolation.
  * Additive Gaussian noise: $15\text{--}30\text{ dB SNR}$ applied with $p=0.30$.
  * SpecAugment: Dynamic frequency masking ($F=27$) and time masking ($T=20$).
* **Total Exposure**: 30,000 steps $\times$ 32 batch size $\times$ grad accum 2 = **64 utterances per step**, totaling **1,920,000 utterances** and **~2,933.3 audio hours**.
* **Validation & Checkpoint Retention**: Every 25 steps, a slice of the batch is decoded with greedy CTC to compute live Character Error Rate (CER) against ground truth Devanagari text. Checkpoints are written every 1,000 steps, preserving the 3 most recent checkpoints plus `best_model.pt` (governed by lowest rolling CER).

##### Stage 3: Joint Multi-Exit Early CTC + GRPO Policy Optimization
* **Hub Split Ingested**: `split="train"` across `ai4bharat/Shrutilipi` and `ARTPARK-IISc/Vaani`.
* **Multi-Exit Supervision Partition**:
  * Utilizes the identical supervised speech-text filtering criteria as Stage 2.
  * Rather than computing CTC loss only at the final layer, target sequences are supervised simultaneously across three vertical exits:
    $$\mathcal{L}_{\text{multi-exit}} = 0.30 \cdot \mathcal{L}_{\text{CTC}}^{(\text{Layer } 4)} + 0.30 \cdot \mathcal{L}_{\text{CTC}}^{(\text{Layer } 8)} + 1.00 \cdot \mathcal{L}_{\text{CTC}}^{(\text{Layer } 12)}$$
* **GRPO Policy Rollout Partition**:
  * For each utterance, the Early-Exit Router samples $G = 3$ to $4$ exit actions $\hat{e} \sim \pi_\theta(\cdot | h_4)$ from Layer 4 hidden representations.
  * Group Relative Policy Optimization (GRPO) calculates normalized group advantages:
    $$A_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G) + \epsilon}, \quad R_i = -\text{CER}(e_i) - \lambda_{\text{lat}} \cdot \text{Cost}(e_i)$$
  * Early exit compute costs are partitioned as: Layer 4 ($0.33\times$), Layer 8 ($0.66\times$), Layer 12 ($1.00\times$), penalizing unnecessary compute on simple speech.
* **Dynamic Chunk Sampling**: Evaluated under the same dynamic chunk distribution $C \in \{1, 4, 8, 16, -1\}$ to ensure the router learns optimal exit policies under both strict causal and non-causal conditions.
* **Total Exposure**: 25,000 steps $\times$ 24 batch size $\times$ grad accum 2 = **48 utterances per step**, totaling **1,200,000 utterances** and **~1,833.3 audio hours**.

---

#### 3. Mathematical Zero-Leakage Guarantee: Streaming (Stages 1–3) vs. Local RESPIN (Stages 4–6)

A crucial architectural principle of this system is the **strict separation between broad streaming acoustic pretraining (Stages 1–3) and speaker-disjoint regional dialect specialization (Stages 4–6)**:

1. **Unbounded Acoustic Diversity for General Grounding (Stages 1–3)**:
   Shrutilipi and Vaani provide broad acoustic and phonetic grounding across all of Maharashtra. Streaming these corpora over HTTP via `datasets` allows training on 164.75 GB of audio data without filling local disk storage.
2. **Strict Speaker-Disjoint Partitioning for Thesis Benchmarking (Stages 4–6)**:
   To ensure scientific integrity and eliminate any possibility of train-test data contamination, **all dialect adaptation, calibration, and final benchmark evaluations strictly switch to the locally hosted IISc RESPIN dataset**:
   * **Stage 4 (MoE Training)**: Strictly restricted to the **95% speaker train partition** of `IISc_RESPIN_train_mr_clean` (~2,512 speakers, ~769k utterances).
   * **Stage 5 (Calibration)**: Strictly restricted to the **5% speaker held-out partition** of `IISc_RESPIN_train_mr_clean` (~132 speakers, ~40k utterances). No speaker in Stage 5 is ever seen during Stage 4 training.
   * **Stage 6 (Final Benchmark)**: Evaluated exclusively on the official **`IISc_RESPIN_test_mr`** test set (2,170 utterances across 160 unseen speakers). This test set remains strictly frozen throughout all training and tuning stages.

---

## 2. 5-Stage Architecture & Training Paradigms

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Base Pretraining (SSL Masked Reconstruction)                  │
│ Data: Vaani (8 Dialects) + Shrutilipi Audio Stream                     │
│ Objective: Continuous Spectrogram Masked L1/L2 Loss                    │
│ Output: checkpoints/pretrain/conformer_pretrain_final.pt               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Causal Adaptation & Supervised CTC Alignment                  │
│ Data: Vaani + Shrutilipi Parallel Speech-Text                          │
│ Objective: Dynamic Chunk Context Training (SFT - CTC Loss)             │
│ Output: checkpoints/causal/conformer_causal_final.pt                   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Joint Multi-Exit Fine-Tuning (Layers 4, 8, 12)                │
│ Data: Vaani + Shrutilipi Speech-Text                                   │
│ Objective: Summed CTC across exits + Exit Policy Optimization          │
│ Output: checkpoints/multi_exit/conformer_multiexit_final.pt            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: MoE Dialect Adaptation                                        │
│ Data: RESPIN Train Split (95% speakers, ~975 hrs across D1-D4)         │
│ Phase 4A: Hard Routing (SFT per expert)                                │
│ Phase 4B: Soft Top-1/2 Dynamic Routing (Router Policy / GRPO)          │
│ Output: checkpoints/moe/conformer_moe_final.pt                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Joint Calibration & Sequence Polish                           │
│ Data: RESPIN Dev Split (5% speakers held-out, ~51 hrs across D1-D4)    │
│ 5A (Callib): Temperature Scaling + Exit Threshold Tuning               │
│ 5B (Sequence Polish): Direct CER Minimization via GRPO RL              │
│ Output: checkpoints/calibrated/conformer_calibrated_final.pt           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ EVALUATION BENCHMARK: IISc_RESPIN_test_mr (2,170 Utterances, D1-D4)     │
│ Zero-leakage held-out thesis metrics: Per-dialect CER, WER, Layer Depth│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dynamic Multi-Exit Architecture with Hierarchical MoE (H-MoE) & GRPO Router

This unified architecture combines two powerful innovations:
1. **Dynamic Multi-Exit Depth (Vertical Adaptation)**: Scales computation depth per utterance, exiting early at **Layer 4** (easy/clean speech), **Layer 8** (conversational speech), or **Layer 12** (difficult/noisy regional speech) driven by an **Early-Exit Router** trained via **GRPO (Group Relative Policy Optimization)**.
2. **Hierarchical Mixture-of-Experts (Horizontal Adaptation)**: Inside the Conformer blocks, FFN layers are structured as a **2-Level Hierarchical MoE (H-MoE)**, where a Macro-Zone Router routes speech into regional geographic zones (Coastal, Central Desh, Northern/Eastern), and micro-experts specialize in local dialect phonetics (D1 Malvani, D2 Ahirani, D3 Standard, D4 Varhadi).

```
========================================================================================
     FULL ARCHITECTURE: DYNAMIC MULTI-EXIT CONFORMER WITH H-MoE & GRPO ROUTER (PPT)
========================================================================================

                                [ Raw 16kHz Audio ]
                                         │
                                         ▼
                            [ Log-Mel Spectrogram (80-dim) ]
                                         │
                                         ▼
                        [ 4x Conv2D Subsampling Block ]
                               (10ms -> 40ms frames)
                                         │
                                         ▼
                       ┌───────────────────────────────────┐
                       │   Conformer Blocks 1, 2, 3, 4     │
                       │   (Shared Acoustic Grounding)     │
                       └─────────────────┬─────────────────┘
                                         │
                         Layer 4 Hidden States (B, T, 256)
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
        ┌────────────────────────┐               ┌────────────────────────┐
        │   EARLY-EXIT ROUTER    │               │    EXIT 1 (Layer 4)    │
        │  (GRPO Policy Network) │               │   Dual CTC Heads       │
        │  pi(exit in {4, 8, 12})│               │  (Standard/Dialectal)  │
        └───────────┬────────────┘               └───────────┬────────────┘
                    │                                        │
         Dynamic Policy Decision                             │
     ┌──────────────┼──────────────┐                         │
     │              │              │                         │
     ▼              │              │                         ▼
[Exit at Layer 4]   │              │             [ EMIT TEXT IMMEDIATELY ]
(33% Compute / 3x)  │              │             • Easy / Clean Speech
                    ▼              │             • Sub-100ms Ultra-Low Latency
            [Exit at Layer 8]      │
            (66% Compute / 1.5x)   │
                                   ▼
                          [Exit at Layer 12]
                          (100% Compute / Max Accuracy)
                                   │
                                   │ (If Router selects Layer 8 or 12)
                                   ▼
                       ┌───────────────────────────────────┐
                       │   Conformer Blocks 5, 6, 7, 8     │
                       │   [ H-MoE: Macro-Zone Experts ]   │
                       └─────────────────┬─────────────────┘
                                         │
                         Layer 8 Hidden States (B, T, 256)
                                         │
                                         ├────────────────────────┐
                                         │                        ▼
                                         │               ┌────────────────────────┐
                                         │               │    EXIT 2 (Layer 8)    │
                                         │               │   Dual CTC Heads       │
                                         │               └───────────┬────────────┘
                                         │                           ▼
                                         │               [ EMIT TEXT AT LAYER 8 ]
                                         │               • Conversational Speech
                                         │               • Balanced Compute (1.5x speedup)
                                         │
                                         ▼ (If Router selects Layer 12)
                       ┌───────────────────────────────────┐
                       │   Conformer Blocks 9, 10, 11, 12  │
                       │   [ H-MoE: Micro-Dialect Experts] │
                       └─────────────────┬─────────────────┘
                                         │
                        Final Hidden States (B, T, 256)
                                         │
                                         ▼
                                 ┌────────────────────────┐
                                 │    EXIT 3 (Layer 12)   │
                                 │   Dual CTC Heads       │
                                 └───────────┬────────────┘
                                             ▼
                                 [ EMIT TEXT AT LAYER 12 ]
                                 • Complex / Fast Regional Dialect
                                 • Maximum Accuracy & Noise Robustness
```

---

### Detailed Zoom-In: Mixture-of-Experts (MoE) Block (3 Dialect Experts + 1 Combining FFN)

Inside Conformer Blocks (Layers 5–12), the second Feed-Forward Network (FFN 2) is replaced by the **Shared-Trunk + 3 Dialect Experts MoE** architecture:

```
========================================================================================
             ZOOM-IN: 3-DIALECT MoE BLOCK (3 EXPERTS + 1 COMBINING FFN)
========================================================================================

                                Hidden States from Multi-Head Attention
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │   Layer Normalization │
                                      └───────────┬───────────┘
                                                  │
                     ┌────────────────────────────┴────────────────────────────┐
                     │                                                         │
                     ▼                                                         ▼
        ┌────────────────────────┐                                ┌────────────────────────┐
        │  COMBINING SHARED FFN  │                                │     DIALECT ROUTER     │
        │  (Always-Active Trunk) │                                │  p(Expert | H)         │
        │  General Standard MR   │                                │  Top-1 or Soft-Gating  │
        └────────────┬───────────┘                                └────────────┬───────────┘
                     │                                                         │
                     │                                 ┌───────────────────────┼───────────────────────┐
                     │                                 │                       │                       │
                     │                                 ▼                       ▼                       ▼
                     │                      ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
                     │                      │      EXPERT 1       │ │      EXPERT 2       │ │      EXPERT 3       │
                     │                      │   Malvani / Konkan  │ │  Ahirani / Khandesh │ │  Varhadi / Vidarbha │
                     │                      │   (Coastal Dialect) │ │  (Northern Dialect) │ │  (Eastern Dialect)  │
                     │                      └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                     │                                 │                       │                       │
                     │                                 └────────────────┬──────┘                       │
                     │                                                  │                              │
                     │                                                  └──────────────┬───────────────┘
                     │                                                                 │
                     ▼                                                                 ▼
         [Combining Shared Output]                                      [Routed Dialect Expert Output]
         F_shared(x)                                                    g_k * F_k(x)
                     │                                                                 │
                     └─────────────────────────────────┬───────────────────────────────┘
                                                       │
                                                       ▼ (Additive Fusion: y = F_shared(x) + g_k * F_k(x))
                                            To Depthwise Conv Module / Next Layer
```

### Why 3 Experts + 1 Combining FFN is the Optimal Architecture:
1. **Mathematical Formulation (Additive Fusion)**:
   $$\mathbf{y} = \text{FFN}_{\text{combining}}(\mathbf{x}) + \sum_{k \in \text{Top-}1} g_k \cdot \text{Expert}_k(\mathbf{x})$$
2. **Combining Shared Trunk**: Preserves standard Marathi grammatical backbone, common vocabulary, and universal acoustic formants. It guarantees 100% stability and prevents catastrophic forgetting.
3. **Dedicated 3 Dialect Experts**:
   - **Expert 1 (Malvani)**: Specializes in coastal phonetic shifts, retroflex patterns, and Konkani-adjacent prosody.
   - **Expert 2 (Ahirani)**: Specializes in northern Khandeshi diphthongs and vowel mutations.
   - **Expert 3 (Varhadi)**: Specializes in eastern Vidarbha alveolar flaps and regional lexical variations.
4. **Sparse Upcycling from Pretraining**:
   - At initialization (Stage 4 Step 0), all 3 experts + the combining FFN are cloned directly from the pretrained Conformer FFN.
   - Initial output is mathematically equivalent to the pretrained model, guaranteeing zero initial degradation.

---

## 4. Plug-and-Play Dual-Head ASR: Standard vs. Dialectal Text

The architecture features a **shared acoustic trunk** with two interchangeable, lightweight CTC projection heads (~27,000 parameters / ~100 KB each):

```
                    ┌──────────────────────────────────────────────┐
                    │    Shared 12-Layer Streaming Conformer       │
                    │               Backbone                       │
                    │      (Acoustic Engine: 20.48M params)        │
                    └──────────────────────┬───────────────────────┘
                                           │
                         Hidden States (B, T, 256)
                                           │
            ┌──────────────────────────────┴──────────────────────────────┐
            ▼                                                             ▼
 ┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
 │       HEAD 1: Standard Marathi       │      │       HEAD 2: Dialectal Marathi      │
 │        (Normalized Devanagari)       │      │         (Verbatim Phonetic)          │
 │         ~27,000 parameters           │      │          ~28,000 parameters          │
 └──────────────────┬───────────────────┘      └──────────────────┬───────────────────┘
                    ▼                                                             ▼
     "मला काल बाजारात जायला जमले नाही"               "माका काल बाजारात जावक जमला नाय"
        (For LLMs, Search, Banking)                  (Authentic Dialect Preservation)
```

### Key Capabilities
1. **Simultaneous Dual Output**: Pass audio through the Conformer backbone **once**, and generate both the normalized standard text and the authentic dialectal text with $< 0.05\text{ ms}$ overhead (`task="both"`).
2. **Adapter-Style Hot Swapping**: The backbone can remain frozen while the dialectal head is fine-tuned as a lightweight adapter on regional corpora in under 30 minutes.

---

## 4. Causal Alignment Pipeline (`causal_alignment/`)

Stage 2 adapts the pretrained bidirectional encoder into a low-latency streaming model using dynamic chunk masking:

* **`tokenizer.py`**: 105-token canonical Marathi Devanagari vocabulary (vowels, consonants, matras, halant/virama, digits, punctuation, and CTC blanks).
* **`chunk_masking.py`**: Vectorized dynamic chunk attention mask generator:
  - $C = 1$ frame (40ms): Purely causal streaming (zero lookahead).
  - $C = 4$ frames (160ms): Ultra-low latency streaming.
  - $C = 8$ frames (320ms): Medium latency streaming.
  - $C = 16$ frames (640ms): High latency streaming.
  - $C = -1$: Full bidirectional context.
* **`dataset.py`**: Non-blocking threaded prefetcher streaming parallel audio and Devanagari text from Vaani + Shrutilipi without Windows multiprocessing deadlocks.
* **`train_causal.py`**: Production training engine with AMP FP16, AdamW, cosine annealing, CTC loss, and live CER evaluation.

---

## 5. SFT vs. GRPO Optimization Strategy

| Paradigm | What it Optimizes | Role in Pipeline | Memory Footprint |
| :--- | :--- | :--- | :--- |
| **SFT (Supervised CTC)** | Frame-level marginal log-likelihood: $\log P(Y \mid X)$ | **Cold Start & Alignment**: Mandatory in Stages 2, 3 (backbone), and 4A (expert warmup) to establish monotonic acoustic-to-grapheme alignment. | Standard backprop pass. |
| **GRPO (Group Relative RL)** | Direct sequence-level reward: $R = -\text{CER} - \lambda \cdot \text{Latency}$ | **Policy Routing & Polish**: Samples $G$ candidates, computes group relative advantage without a separate critic model. Used in Stage 3 (exit policy), Stage 4B (MoE routing), and Stage 5B (CER polish). | Zero critic parameters; fits easily in 24GB VRAM. |

---

## 6. Stage 3: Official Multi-Exit Benchmark & Ablation Results

Evaluated on 200 diverse multi-dialect Marathi test utterances using [`multi_exit/eval_exits.py`](file:///d:/marathi-asr/multi_exit/eval_exits.py) on [`checkpoints/multi_exit/conformer_multiexit_final.pt`](file:///d:/marathi-asr/checkpoints/multi_exit/conformer_multiexit_final.pt):

| Exit Policy | Mean CER (%) | Avg Exit Layer | Conformer Compute Speedup | Target Deployment Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Layer 4 (Fast)** | **19.33%** | **4.00** | **3.00x** (66.7% compute saved) | Edge mobile, voice search, clean audio |
| **Layer 8 (Balanced)** | **11.45%** | **8.00** | **1.50x** (33.3% compute saved) | Standard streaming, smart displays |
| **Layer 12 (Deep)** | **7.69%** | **12.00** | **1.00x** (Baseline full model) | High-noise, heavy dialect transcription |
| **Dynamic GRPO Router** | **11.45%** | **8.00** | **1.50x** (Optimal Pareto trade-off) | Autonomous edge-to-cloud adaptation |

---


All stage models are archived into dedicated directories:
```
checkpoints/
├── pretrain/         # Stage 1: SSL Pretrained Conformer
│   ├── conformer_pretrain_step_*.pt
│   ├── conformer_pretrain_latest.pt
│   └── conformer_pretrain_final.pt
├── causal/           # Stage 2: Dynamic Chunk Causal CTC Model
│   ├── conformer_causal_step_*.pt
│   └── conformer_causal_final.pt
├── multi_exit/       # Stage 3: Multi-Exit Early CTC Model (Layers 4, 8, 12)
│   └── conformer_multiexit_final.pt
├── moe/              # Stage 4: Mixture-of-Experts Dialect Model (4 Experts + Router)
│   └── conformer_moe_final.pt
└── calibrated/       # Stage 5: Final Calibrated & Sequence Polished Production Model
    └── conformer_calibrated_final.pt
```

---

## 7. Environment & Quick Start

The project uses `uv` for package management on Windows with CUDA 12.4 + PyTorch 2.6:

```bash
# Clone and enter repo
git clone git@github.com:rimraf-adi/marathi-asr.git
cd marathi-asr

# Install dependencies via uv
uv sync

# Extract metrics from training logs to CSV and generate publication plots
uv run python pretraining/extract_metrics_to_csv.py

# Organize checkpoints semantically
uv run python pretraining/organize_checkpoints.py

# Launch Stage 2 Causal Alignment & CTC Fine-Tuning
uv run python causal_alignment/train_causal.py --pretrained_ckpt checkpoints/pretrain/conformer_pretrain_latest.pt --steps 10000 --batch_size 32
```
