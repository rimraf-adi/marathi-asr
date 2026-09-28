# Research Report: Pure Decoder-Only ASR (No Encoder Backbone)

## Executive Summary
Traditional Automatic Speech Recognition (ASR) systems rely on either **hybrid encoder-decoder** architectures (e.g., OpenAI Whisper, SpeechT5) where continuous audio features are processed by an encoder and attended to via cross-attention, or **encoder-only CTC/Transducer** systems (e.g., Conformer-CTC, Zipformer). 

In contrast, **pure decoder-only ASR** treats speech recognition as an autoregressive next-token prediction task—identically to how causal LLMs (like GPT-4, LLaMA, or Mistral) process text. In this paradigm:
1. Continuous audio is discretized into a sequence of discrete audio tokens via an offline or frozen audio tokenizer.
2. The discrete audio tokens and text tokens share a unified embedding space in a single causal Transformer decoder.
3. No cross-attention layers, audio encoder backbones, or separate acoustic feature extractors exist inside the model graph during inference. Speech recognition is performed simply by prompting the decoder with audio tokens and autoregressively sampling text tokens until `<eos>`.

---

## 1. Audio Tokenization for Decoder-Only ASR

Feeding audio into a standard causal transformer requires converting continuous raw waveforms $x \in \mathbb{R}^T$ into discrete token IDs $a_t \in \{1, \dots, K\}$.

```
Raw Audio (16/24 kHz)
        │
        ▼
[Audio Tokenizer] (HuBERT + K-Means / RVQ Codec / SpeechTokenizer / Whisper-VQ)
        │
        ▼
Discrete Audio Tokens: [a₁, a₂, a₃, ..., a_N] (Vocabulary size: K)
        │
        ▼
[Causal Transformer Decoder] (Unified Embedding Matrix: Text Vocab + Audio Vocab)
        │
        ▼ Autoregressive Generation
Output Text Tokens: [t₁, t₂, ..., t_U]
```

---

### 1.1 HuBERT k-Means Clustering

* **Mechanism**: HuBERT (Hidden-Unit BERT) processes audio through a 7-layer temporal convolutional encoder (stride 320 at 16 kHz = 20 ms frame rate / 50 Hz), followed by BERT-like Transformer blocks. Continuous latent hidden representations $h_t \in \mathbb{R}^D$ from a selected transformer layer are extracted and quantized using offline **k-means clustering**.
* **Layer Selection**:
  * **HuBERT-Base (12 layers, 768-d)**: 
    * *Layer 6*: Often used in early speech language models (e.g., GSLM). Captures fine-grained acoustic and phonetic information, but retains significant speaker timbre/pitch bias.
    * *Layer 9 / Layer 11*: Captures phonemic and pseudo-lexical semantics with reduced acoustic/speaker variability. Layer 9 was the clustering target for HuBERT iteration 2.
  * **HuBERT-Large (24 layers, 1024-d)**:
    * *Layer 18*: Standard in Meta's TWIST and Textless NLP. Balances phonetic detail and invariant linguistic representation.
    * *Layer 22*: Used in SpiRit-LM and LauraGPT. Highly semantic, stripping away speaker-specific acoustic details.
* **Cluster Count ($K$)**:
  * $K = 100$: Used in early GSLM. Provides compact vocabulary but suffers from phonetic ambiguity and high quantization error.
  * $K = 500$: Standard sweet spot (e.g., SpiRit-LM, TWIST). Well-aligned with phonemic inventories across accents.
  * $K = 1000$ to $2048$: Used in SpeechGPT (1000 units from mHuBERT) and AudioPaLM (1024 units from w2v-BERT). Finer acoustic resolution, lower distortion.
* **Deduplication (Run-Length Deduplication)**:
  * *Raw*: 50 tokens/sec. Speech has high inertia; a 200 ms vowel produces 10 identical consecutive tokens (`[42, 42, 42, 42, 42, 42, 42, 42, 42, 42]`).
  * *Deduplicated*: Consecutive identical tokens are collapsed to a single token (`[42, 42, 42, ...] -> [42]`).
  * *Impact on ASR*: Highly effective for ASR! Deduplication compresses token rates from **50 tokens/sec down to ~15–25 tokens/sec** (~2x–3x compression). Since self-attention has $O(N^2)$ memory and compute complexity, halving the sequence length provides a ~4x reduction in prefill attention cost without loss of linguistic content. (Note: While deduplication causes loss of prosody/duration for TTS, text transcription needs no duration information).

---

### 1.2 EnCodec and SoundStream Neural Audio Codecs

* **Mechanism**: Neural audio codecs employ an encoder-decoder architecture with **Residual Vector Quantization (RVQ)**.
* **Residual Vector Quantization (RVQ)**:
  * Instead of searching a single exponential codebook ($K^{N_q}$), RVQ cascades $N_q$ quantizers.
  * Quantizer 1 quantizes continuous embedding $z \to q_1$. The residual $r_1 = z - q_1$ is passed to Quantizer 2 $\to q_2$, and so on:
    $$z \approx \sum_{j=1}^{N_q} q_j$$
* **Codebook Sizes**:
  * Typically $K = 1024$ entries ($10$ bits per codebook) per quantizer stage.
  * Number of codebooks: $N_q \in \{4, 8, 12, 16, 32\}$.
* **Frame Rates & Bitrates**:
  * **EnCodec (24 kHz)**: Strides $2 \times 4 \times 5 \times 8 = 320$. Frame rate = $24000 / 320 = 75$ frames/sec (75 Hz). At 8 codebooks (6 kbps), this produces $75 \times 8 = \mathbf{600\text{ tokens/sec}}$.
  * **EnCodec (16 kHz)**: Downsampling factor 320. Frame rate = $16000 / 320 = 50$ frames/sec (50 Hz). At 4 codebooks (2 kbps), produces $50 \times 4 = \mathbf{200\text{ tokens/sec}}$; at 8 codebooks (4 kbps), $\mathbf{400\text{ tokens/sec}}$.
  * **SoundStream (16 kHz)**: 50 Hz frame rate. At 12 codebooks (6 kbps), produces $\mathbf{600\text{ tokens/sec}}$.
* **Limitation for Decoder-Only ASR**:
  * Acoustic codecs optimize for full waveform reconstruction (phase, timbre, room acoustics, background noise).
  * Ingesting 600 tokens/sec causes massive sequence bloat: a 10-second audio clip requires 6,000 tokens in the transformer context window!
  * Quantizers $2 \dots N_q$ encode fine acoustic residuals that are irrelevant for text transcription and act as noise.

---

### 1.3 SpeechTokenizer (Semantic + Acoustic Disentanglement)

Zhang et al. (2023) developed **SpeechTokenizer** specifically to resolve the tension between semantic SSL units (HuBERT) and acoustic RVQ codecs (EnCodec).

* **Architecture**: RVQ-based convolutional codec with 8 quantizers operating at **50 Hz** (16 kHz audio downsampled by 320).
* **Hierarchical Disentanglement**:
  * **Quantizer 1 (RVQ-1)**: Enforced via a distillation loss to match the continuous representations of **HuBERT (Layer 9/11)**.
    $$\mathcal{L}_{\text{semantic}} = 1 - \cos(\mathbf{e}_{\text{RVQ-1}}, \mathbf{h}_{\text{HuBERT}})$$
  * **Quantizers 2–8**: Optimized solely via adversarial, mel-reconstruction, and commitment losses to capture residual acoustic details (speaker identity, pitch, environment).
* **Application to Pure Decoder-Only ASR**:
  * The ASR decoder **only ingests the Quantizer 1 (RVQ-1) tokens** (codebook size $K = 1024$), completely discarding stages 2–8!
  * Yields a clean token rate of **50 tokens/sec** with high semantic purity.

---

### 1.4 Whisper Encoder Features Quantized via VQ (Whisper-VQ)

* **Mechanism**: OpenAI Whisper's encoder processes 80/128-channel log-mel filterbanks at 100 Hz, applies two 1D convolutions with stride 2 (downsampling to **50 Hz**), followed by Transformer encoder blocks pre-trained on 680,000 hours of weakly supervised speech.
* **Vector Quantization**:
  * Projects Whisper's continuous encoder vectors (e.g., 768-d for Base, 1280-d for Large) through an `RQBottleneckTransformer` or standard VQ codebook (typically $K = 512, 1024, \text{or } 4096$).
  * Models like **WhisperSpeech / WhisperVQ** (Jan-HQ / Collabora) introduce an additional strided pooling layer to downsample from 50 Hz to **25 Hz** (25 tokens/sec).
* **Advantage for ASR**:
  * Whisper's encoder is trained with a direct transcription objective. Its latent space is explicitly structured around linguistic tokens and is robust to noise and accents.
  * Quantizing Whisper encoder features yields discrete tokens with superior phonemic and semantic alignment compared to unsupervised codecs.

---

### 1.5 Token Rates & Characteristics Summary

| Tokenizer | Frame Rate | Codebooks / Hierarchy | Effective Token Rate | Primary Information Encoded |
|---|---|---|---|---|
| **HuBERT k-means (raw)** | 50 Hz | 1 codebook ($K=100 - 2048$) | **50 tokens/sec** | Phonetic / Semantic |
| **HuBERT k-means (dedup)** | Dynamic | 1 codebook ($K=500 - 1000$) | **~15–25 tokens/sec** | Lexical / Phonemic progression |
| **EnCodec (16 kHz, 3 kbps)** | 50 Hz | 4 RVQ codebooks ($K=1024$) | **200 tokens/sec** | Acoustic + Waveform detail |
| **EnCodec (24 kHz, 6 kbps)** | 75 Hz | 8 RVQ codebooks ($K=1024$) | **600 tokens/sec** | Full audio / Timbre / Background |
| **SoundStream (16 kHz)** | 50 Hz | 12 RVQ codebooks ($K=1024$) | **600 tokens/sec** | Acoustic fidelity |
| **SpeechTokenizer (Full)** | 50 Hz | 8 RVQ layers ($K=1024$) | **400 tokens/sec** | L1: Semantic; L2–8: Acoustic |
| **SpeechTokenizer (L1 Only)**| 50 Hz | 1 codebook ($K=1024$) | **50 tokens/sec** | Pure Phonetic / Semantic |
| **Whisper-VQ (Standard)** | 50 Hz | 1 codebook ($K=512 - 4096$) | **50 tokens/sec** | Strongly Lexical / Semantic |
| **Whisper-VQ (Downsampled)**| 25 Hz | 1 codebook ($K=512 - 4096$) | **25 tokens/sec** | Strongly Lexical / Semantic |
| **Text BPE (Reference)** | N/A | Subwords ($V = 32k - 128k$) | **~3–4 tokens/sec** | High-level semantics / Syntax |

---

## 2. Pure Decoder-Only ASR Models (No Separate Encoder)

A **pure decoder-only** architecture contains **zero encoder modules** inside the model graph. Audio tokens are embedded through a lookup table and concatenated directly into the causal self-attention context.

```
       Causal Self-Attention Transformer Decoder (e.g., LLaMA, PaLM, OPT)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   a₁    a₂    a₃    ...    a_T    [ASR]    t₁        t₂        ...    [EOS]     │
│   ▲     ▲     ▲            ▲       ▲       ▲         ▲                 ▲        │
└───┼─────┼─────┼────────────┼───────┼───────┼─────────┼─────────────────┼────────┘
    │     │     │            │       │       │         │                 │
┌───┴─────┴─────┴────────────┴───────┴───────┴─────────┴─────────────────┴────────┐
│ Unified Embedding Matrix W_e ∈ ℝ^( (|V_text| + |V_audio|) × D )                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 SpeechGPT (Zhang et al., 2023)
* **Backbone**: LLaMA-7B / LLaMA-13B causal decoder.
* **Tokenization**: Discrete speech units derived from **mHuBERT-147** clustered into **1,000 units** via k-means, followed by **run-length deduplication**.
* **Vocabulary Construction**: Augmented LLaMA's 32,000 text vocabulary with 1,000 speech tokens `<s_0>` through `<s_999>`:
  $$|V_{\text{total}}| = 32,000 + 1,000 = 33,000$$
* **ASR Formulation**: Formulated as instruction tuning:
  ```
  Human: <s_234><s_89><s_12>... Transcribe the speech to text.
  Assistant: The quick brown fox jumps over the lazy dog.
  ```

### 2.2 SpiRit-LM (Meta, 2024)
* **Backbone**: LLaMA-2 7B architecture.
* **Variants**:
  * **SpiRit-LM Base**: Integrates HuBERT phonetic tokens (500 clusters, 25 Hz / 50 Hz).
  * **SpiRit-LM Expressive**: Interleaves phonetic units, pitch tokens (VQ-quantized $F_0$), and style/speaker tokens.
* **Training Style**: Trained on naturally interleaved speech and text at the word and sentence levels. Can perform ASR natively in-context without task-specific adapters.

### 2.3 TWIST: Textually Warm-Initialized Speech Transformer (Hassid et al., NeurIPS 2023)
* **Backbone**: Pretrained OPT (125M to 13B) and LLaMA models.
* **Concept**: Investigates whether speech language models should be trained from scratch (cold-start) or warm-started from pretrained text models.
* **Architecture**: The text embedding table is replaced or augmented with discrete speech tokens derived from **mHuBERT-base-25Hz** (500 clusters).
* **Key Finding**: Initializing from a text LLM provides a massive boost in sample efficiency, acoustic-to-linguistic binding, and grammatical fluency over cold-started models.

### 2.4 AudioLM (Borsos et al., 2023) & AudioPaLM (Rubenstein et al., 2023)
* **AudioLM**: Pure decoder-only hierarchical modeling using semantic tokens (w2v-BERT, 1024 clusters) and acoustic tokens (SoundStream, 12 RVQ stages).
* **AudioPaLM**: Unifies text (PaLM-2 SentencePiece) and speech (w2v-BERT semantic tokens + SoundStream acoustic tokens) in a single decoder-only PaLM-2 model.
  * For ASR, the input is purely semantic tokens ($w2v\text{-BERT}$) and the target is text tokens.
  * Outperformed hybrid Whisper models on multi-lingual speech-to-text translation (AST) and ASR benchmarks.

### 2.5 VoxtLM (Maiti, Watanabe et al., ICASSP 2024 / ESPnet)
* **Backbone**: Decoder-only Transformer (130M, 350M, 1B parameters) trained from scratch or warm-started.
* **Task Conditioning**: Uses task prompt tokens to multiplex 4 capabilities in one model:
  * ASR: `<|task_asr|> <|audio|> a₁ a₂ ... a_T <|text|> t₁ t₂ ... t_U <|eos|>`
  * TTS: `<|task_tts|> <|text|> t₁ ... t_U <|audio|> a₁ ... a_T <|eos|>`
  * Speech continuation and text continuation.

### 2.6 LauraGPT (Xie et al., Alibaba 2023)
* **Backbone**: Causal GPT decoder.
* **Tokenization**: Integrates continuous audio embeddings as prefix features and discrete codec tokens (FunCodec) for generation. Demonstrates competitive WER on LibriSpeech compared to hybrid encoder-decoder baselines.

---

## 3. Training Decoder-Only ASR

### 3.1 Training Data Formatting

Decoder-only ASR formats input audio and target text as a single contiguous sequence of tokens:

#### Format A: Direct Transduction Prompting (Standard ASR SFT)
```
[BOS] <|audio_start|> a_14 a_892 a_43 ... a_102 <|audio_end|> <|task_asr|> <|text_start|> The weather is sunny [EOS]
```

#### Format B: Instruction-Tuned Conversational Format (SpeechGPT)
```
<|im_start|>user
[speech_token_34][speech_token_881]... Please transcribe the above speech into text.<|im_end|>
<|im_start|>assistant
Please transcribe the above speech into text.<|im_end|>
```

#### Format C: Word-Level Interleaving (SpiRit-LM)
Used during foundation pretraining to teach the decoder tight temporal cross-modal alignment:
```
<|speech|> a_1..a_8 <|text|> "Hello" <|speech|> a_9..a_24 <|text|> "world"
```

---

### 3.2 Loss Function: Target-Only Cross-Entropy (Prompt Masking)

During ASR training, the causal language modeling loss is **masked out for audio tokens**. The model must not waste capacity predicting audio tokens during transcription fine-tuning.

Let sequence $X = [x_1, \dots, x_M]$ consist of prefix audio prompt tokens $\mathbf{a}_{1:T}$ and target text tokens $\mathbf{t}_{1:U}$ ($M = T + U$). The training objective is:

$$\mathcal{L}_{\text{ASR}}(\theta) = - \sum_{u=1}^{U} \log P_\theta(t_u \mid \mathbf{a}_{1:T}, t_{<u})$$

In implementation (e.g., PyTorch / HuggingFace):
```python
# Audio token positions are assigned label = -100
labels = [-100] * len(audio_tokens) + text_tokens[1:] + [eos_token_id]
loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
```

#### Multi-Task / Joint Pre-Training Loss:
If pretraining jointly on speech modeling, text modeling, and ASR:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \lambda_{\text{audio}} \mathcal{L}_{\text{audio}} + \lambda_{\text{ASR}} \mathcal{L}_{\text{ASR}}$$

---

### 3.3 Text-Only Pretraining: Warm-Start vs. Cold-Start

| Property | Cold-Start (From Scratch) | Warm-Start (Pretrained LLM: LLaMA/Mistral) |
|---|---|---|
| **Linguistic Knowledge** | Must learn grammar, spelling, facts from speech | Inherits rich syntax, world knowledge, and vocabulary |
| **Sample Efficiency** | Very low (requires >10,000+ hours of audio) | High (converges with 1,000–3,000 hours) |
| **Hallucination Rate** | Low hallucination, but high phonetic spelling errors | Rare spelling errors, but can hallucinate plausible text |
| **Convergence Speed** | Slow (hundreds of thousands of steps) | 5x–10x faster convergence |

#### Vocabulary Expansion Procedure for Warm-Starting:
1. Preserve existing text tokenizer vocabulary $V_{\text{text}}$ of size $N$.
2. Append $K$ discrete audio tokens: $V_{\text{new}} = V_{\text{text}} \cup \{ \langle a_0 \rangle, \dots, \langle a_{K-1} \rangle \}$.
3. Resize the decoder's embedding matrix $W_e \in \mathbb{R}^{(N+K) \times D}$ and language model head $W_{\text{lm}} \in \mathbb{R}^{(N+K) \times D}$.
4. Initialize the new $K$ rows using $\mathcal{N}(0, \sigma^2)$ where $\sigma$ matches the standard deviation of existing text embeddings.
5. **Modality Adaptation Phase**: Freeze the transformer layers and train *only* the new embedding rows on audio data for 5,000–10,000 steps, before unfreezing with LoRA or full fine-tuning.

---

### 3.4 Data Scale Requirements

* **Cold-start from scratch**:
  * LibriSpeech 960h is **insufficient** for a pure decoder-only model (yields high WER > 15%).
  * Minimum **10,000 to 50,000 hours** (e.g., Libri-Light 60k hours) required for competitive performance.
* **Warm-started from 7B text LLM**:
  * With HuBERT / SpeechTokenizer units: **1,000 to 5,000 hours** of paired speech-text data is sufficient.
  * With **Whisper-VQ tokens**: As little as **200 to 1,000 hours** achieves strong ASR performance because Whisper features are already semantically organized.

---

## 4. Inference & Decoding

### 4.1 Autoregressive Decoding Mechanics

```
Step 1 (Prefill): 
Input:  [a₁, a₂, ..., a_T, <task_asr>]  ──> Forward Pass ──> Compute & Save KV Cache
Output: Logits for first text token t₁   ──> Select t₁ (Greedy / Beam Search)

Step 2 (Decode Token 1):
Input:  [t₁] (with cached KV)          ──> Single Step Forward ──> Append to KV Cache
Output: Logits for t₂                  ──> Select t₂

... Repeat until t_u = <eos>
```

---

### 4.2 KV-Cache Mechanics

* **Prefill Phase**: All $T$ audio tokens are processed simultaneously in parallel through causal self-attention. Keys ($K$) and Values ($V$) across all layers are computed and stored in memory.
* **Generation Phase**: For each text token $u \in \{1, \dots, U\}$:
  * Only 1 new token vector is passed into the model.
  * Attention is computed by multiplying the new query $Q_u$ against cached keys $[K_{\text{audio}}; K_{\text{text}, <u}]$.
  * Complexity per step is $O(T + u)$ instead of $O((T + u)^2)$.
* **Memory Footprint**:
  $$\text{Memory}_{\text{KV}} = 2 \times L_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times (T_{\text{audio}} + U_{\text{text}}) \times \text{bytes\_per\_elem}$$
  * *Example*: 10 seconds of speech with raw HuBERT (500 tokens) + 50 text tokens in a 7B model ($L=32, H_{\text{kv}}=32, D_{\text{head}}=128$, float16):
    $$\text{Memory}_{\text{KV}} = 2 \times 32 \times 32 \times 128 \times 550 \times 2 \approx 288\text{ MB per stream}$$

---

### 4.3 Beam Search vs. Greedy vs. Nucleus Sampling

* **Why Nucleus Sampling ($p < 1.0$) Fails in ASR**:
  * ASR is an exact acoustic transcription task, not creative text completion.
  * Sampling introduces stochastic substitution errors, word omissions, repeated stutters, and hallucinations.
* **Greedy Search ($T = 0$)**:
  * Selects $t_u = \arg\max_v P(v \mid \mathbf{a}_{1:T}, t_{<u})$.
  * Default choice for production systems. Fast, deterministic, and avoids repetitive looping when paired with repetition penalties.
* **Beam Search (Beam Width 3–5)**:
  * Maintains top $B$ hypotheses.
  * Provides a **0.2%–0.8% absolute WER reduction** over greedy decoding, but increases compute and memory usage by $B\times$.

---

### 4.4 Speed & Efficiency: Decoder-Only ASR vs. CTC

| Dimension | CTC (e.g., Conformer-CTC) | Pure Decoder-Only ASR (e.g., LLaMA-Speech) |
|---|---|---|
| **Architecture** | Encoder-Only (Continuous audio $\to$ Linear $\to$ CTC) | Causal Decoder-Only (Audio tokens $\to$ Text tokens) |
| **Decoding Style** | **Non-autoregressive** (All frames in parallel) | **Autoregressive** ($U$ sequential token steps) |
| **Decoding Steps** | $O(1)$ forward pass | $O(U)$ sequential forward passes |
| **Inference Latency** | **10–40 ms** (Real-Time Factor: $0.005 - 0.02$) | **200–800 ms** (Real-Time Factor: $0.05 - 0.3$) |
| **GPU Bottleneck** | Compute-bound (matrix multiplications in parallel) | Memory bandwidth-bound (sequential KV-cache fetching) |
| **Language Modeling** | Weak conditional independence; needs external LM | Deep, intrinsic LLM reasoning, grammar, and context |
| **Hallucination Risk** | **Zero hallucination** (strictly aligned to audio) | Risk of plausible text hallucination under noise |
| **Model Size** | 30M – 600M parameters | 1B – 13B parameters |

> **Key Takeaway**: CTC is **10x to 50x faster** with lower latency and zero hallucination. However, pure decoder-only ASR excels in complex contextual biasing, zero-shot entity correction, multi-turn conversational understanding, and unified speech-to-text-to-speech architectures.

---

## 5. Key Papers, Models, and Open-Source Repositories

### Foundational Papers & Architectures
1. **GSLM (Generative Spoken Language Model)**
   * *Citation*: Lakhotia et al., "Generative Spoken Language Modeling from Raw Audio", TACL 2021.
   * *Contribution*: Pioneered textless speech LM using HuBERT k-means discrete units.
2. **AudioLM**
   * *Citation*: Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation", IEEE/ACM TASLP 2023.
   * *Contribution*: Introduced the semantic-to-acoustic hierarchical discrete token modeling framework.
3. **AudioPaLM**
   * *Citation*: Rubenstein et al., "AudioPaLM: A Large Language Model That Can Speak and Listen", arXiv:2306.12925, Google, 2023.
   * *Contribution*: Decoder-only unified PaLM model performing ASR, AST, and TTS with discrete audio tokens.
4. **SpeechGPT & SpeechTokenizer**
   * *Citation*: Zhang et al., "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities", EMNLP 2023.
   * *Citation*: Zhang et al., "SpeechTokenizer: Unified Speech Tokenizer for Speech-Language Models", ICLR 2024.
   * *Contribution*: First open-source speech instruction-tuned LLM and disentangled RVQ semantic tokenizer.
5. **TWIST**
   * *Citation*: Hassid et al., "Textually Pretrained Speech Language Models", NeurIPS 2023.
   * *Contribution*: Formalized warm-starting speech transformers from text LLMs; established scaling laws.
6. **SpiRit-LM**
   * *Citation*: Meta AI, "SpiRit-LM: Interleaved Spoken and Written Language Model", ACL 2024.
   * *Contribution*: Word-level speech-text interleaving for unified multimodal foundation models.
7. **VoxtLM**
   * *Citation*: Maiti, Peng, Watanabe et al., "VoxtLM: Unified Decoder-Only Models for Consolidating Speech Recognition, Synthesis and Continuation", ICASSP 2024.
   * *Contribution*: ESPnet-based pure decoder-only multi-task ASR/TTS/LM framework.

### Open-Source Implementations & Code Repositories
* **SpeechTokenizer**: [https://github.com/ZhangXuanji/SpeechTokenizer](https://github.com/ZhangXuanji/SpeechTokenizer)
* **SpeechGPT**: [https://github.com/0nutation/SpeechGPT](https://github.com/0nutation/SpeechGPT)
* **SpiRit-LM**: [https://github.com/facebookresearch/spiritlm](https://github.com/facebookresearch/spiritlm)
* **TWIST (TextlessLocal)**: [https://github.com/facebookresearch/textlesslocal](https://github.com/facebookresearch/textlesslocal)
* **WhisperSpeech / WhisperVQ**: [https://github.com/collabora/WhisperSpeech](https://github.com/collabora/WhisperSpeech)
* **AudioLM PyTorch Implementation**: [https://github.com/lucidrains/audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch)
* **ESPnet (VoxtLM Recipe)**: [https://github.com/espnet/espnet](https://github.com/espnet/espnet)
