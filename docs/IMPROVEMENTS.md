# IMPROVEMENTS.md — Comprehensive Technical Audit & Improvement Plan

**Project**: Multi-Dialect Marathi Streaming Conformer ASR  
**Audit Date**: 2026-09-20  
**Scope**: Every source file, all metric CSVs, all training logs, complete architecture review  
**Auditor Methodology**: Read every `.py` file, every CSV, every log, before writing a single line

---

## Table of Contents

1. [Critical Bugs — Things That Are Actively Wrong](#1-critical-bugs--things-that-are-actively-wrong)
2. [Architectural Design Flaws](#2-architectural-design-flaws)
3. [Training Methodology Issues](#3-training-methodology-issues)
4. [Data Pipeline Problems](#4-data-pipeline-problems)
5. [Metric & Evaluation Integrity Issues](#5-metric--evaluation-integrity-issues)
6. [Code Quality & Engineering Debt](#6-code-quality--engineering-debt)
7. [Prioritized Improvement Roadmap](#7-prioritized-improvement-roadmap)

---

## 1. Critical Bugs — Things That Are Actively Wrong

### 1.1 CRITICAL: GRPO Router Has Completely Collapsed (Stage 3)

**Evidence**: `telemetry/csvs/stage3_multiexit_metrics.csv`

The last **200+ logged steps** show:
```
prob_l4 = 0.0, prob_l8 = 1.0, prob_l12 = 0.0
```

The GRPO Early-Exit Router has **mode-collapsed to always select Layer 8**. It is not making dynamic routing decisions — it is a dead component. This means:
- The "Dynamic GRPO Router" benchmark result (11.45% CER, 1.50x speedup) is **just Layer 8 performance masquerading as a learned policy**.
- The router learned a degenerate fixed policy: lambda_latency=0.25 makes Layer 8 always dominate (decent accuracy, moderate latency), so the router converged to a point mass.

**Root Cause** (multi_exit/grpo_loss.py lines 43-60):
The reward function `R = (1 - CER) - 0.25 * (layer / 12)` has a structural bias. With Layer 8 producing CER ~11% and Layer 12 producing CER ~8%, the reward calculates:
- Layer 8: `(1 - 0.11) - 0.25 * (8/12) = 0.89 - 0.167 = 0.723`
- Layer 12: `(1 - 0.08) - 0.25 * (12/12) = 0.92 - 0.25 = 0.670`
- Layer 4: `(1 - 0.19) - 0.25 * (4/12) = 0.81 - 0.083 = 0.727`

Layers 4 and 8 have **nearly identical rewards**, but the group normalization + small advantage magnitudes cause the policy gradient to be extremely noisy for distinguishing them. Over 10k steps, the policy settles into the deterministic mode with lowest variance: always pick Layer 8.

**Fix Required**:
- Add an **entropy bonus** to the GRPO loss (`H_bonus = -alpha * sum(pi * log(pi))`) to prevent premature collapse.
- Anneal the latency penalty from 0 to 0.25 over training so the router first learns accuracy, then trades off.
- Consider temperature scaling on the router softmax.

---

### 1.2 CRITICAL: MoE Layer Phase 4B Expert Forward Is Manually Unrolled and Drops the Residual

**File**: moe/moe_layer.py lines 114-125

In Phase 4B (dynamic routing), the expert forward is implemented as manual unrolled operations:
```python
exp_out = self.experts[expert_id].linear1(self.experts[expert_id].layer_norm(expert_inp))
exp_out = self.experts[expert_id].activation(exp_out)
exp_out = self.experts[expert_id].dropout1(exp_out)
exp_out = self.experts[expert_id].linear2(exp_out)
exp_out = self.experts[expert_id].dropout2(exp_out)
flat_expert_out[mask] = 0.5 * exp_out * flat_top1_weights[mask]
```

But `FeedForwardModule.forward()` does:
```python
return residual + 0.5 * x  # Macaron half-step residual
```

**The manual unrolling skips the residual connection.** The expert is computing `0.5 * FFN(x)` without the `residual + ...` part. This means the expert output is a **pure transformed projection** rather than a residual perturbation. It effectively changes the semantics of the expert computation entirely and removes the identity shortcut that stabilizes gradient flow.

Additionally, it applies the `0.5` scale factor outside the standard Macaron convention, creating an inconsistency with how the combining FFN trunk computes its output (which uses the standard `residual + 0.5 * FFN(x)` path).

**Fix Required**: Call `self.experts[expert_id](expert_inp)` directly instead of manually unrolling. The `FeedForwardModule.forward()` already handles the half-step residual correctly. Then subtract the input to get just the delta: `expert_delta = self.experts[expert_id](expert_inp) - expert_inp`, and weight by `flat_top1_weights`.

---

### 1.3 CRITICAL: Double log_softmax in Stage 4 & Stage 5 Training

**Files**: moe/train_moe.py lines 149-150, calibration/train_calibration.py lines 152-153

```python
logits_l12 = ctc_dict["exit_log_probs"][12]
log_probs_l12 = logits_l12.log_softmax(dim=-1).transpose(0, 1)
```

But `ctc_dict["exit_log_probs"]` already returns `log_softmax` output from `CTCHead.forward()` (model/heads.py lines 67-69):
```python
logits = self.proj(x)
return torch.log_softmax(logits, dim=-1)
```

**This applies `log_softmax` twice**, computing `log(softmax(log(softmax(x))))`. The second log_softmax squashes already-negative log-probabilities further toward zero, distorting the gradient landscape for CTC loss. It does not cause NaN (CTC uses `zero_infinity=True`), but it makes training significantly less efficient — the model has to fight against distorted gradients.

Note: Stage 2's train_causal.py does NOT have this bug — it correctly uses the output directly and transposes.

**Fix**: Remove the `.log_softmax(dim=-1)` call in train_moe.py and train_calibration.py. Just transpose the already-log_softmaxed output.

---

### 1.4 IMPORTANT: Stage 5 Calibration Uses `tokenizer.decode()` Instead of `tokenizer.ctc_decode()`

**File**: calibration/train_calibration.py line 178

```python
pred_text = tokenizer.decode(pred_tokens)
```

`decode()` just maps IDs to characters without CTC blank-collapsing. The predictions will contain massive character repetitions (e.g., `maaazhhyaa` instead of `mazya`). This means the **CER being logged during Stage 5 training is artificially inflated** and does not reflect the model's actual decoded output quality. The logged CER is likely 2-3x higher than reality.

**Fix**: Replace with `tokenizer.ctc_decode(pred_tokens)`.

---

## 2. Architectural Design Flaws

### 2.1 IMPORTANT: No SpecAugment Applied During Supervised Training (Stages 2-5)

The `SpecAugment` module exists in model/masking.py and is instantiated in `StreamingASRModel.__init__()`, but it is **never called** in `forward_ctc()`. SpecAugment is only used during SSL pretraining.

This is a major regularization gap. For a 22M-parameter model trained on ~1,000 hours of speech, SpecAugment is considered **mandatory** by every modern ASR system (Conformer paper, Whisper, IndicConformer). Without it, the model overfits to specific frequency patterns and speaker acoustics.

**Fix**: Add `spec = self.spec_augment(spec)` after `spec = self.frontend(waveforms)` in `forward_ctc()`, guarded by `self.training`.

---

### 2.2 Conv2dSubsampling4 Uses nn.ReLU Instead of nn.GELU/nn.SiLU

**File**: model/conformer.py lines 47-52

The subsampling block uses `ReLU`, while the rest of the Conformer uses `SiLU` (Swish). This is a minor inconsistency, but Swish has been shown to improve ASR convergence in subsampling blocks (see ESPnet Conformer and WeNet implementations).

---

### 2.3 ConformerConvModule Uses GroupNorm(1, d_model) Instead of BatchNorm1d

**File**: model/conformer.py line 132

The original Conformer paper specifies BatchNorm1d inside the convolution module. `GroupNorm(1, ...)` is effectively LayerNorm over the channel dimension, which has different normalization statistics (per-sample vs. per-batch). For streaming/causal inference this can be a deliberate choice, but it should be documented.

---

### 2.4 CTC Head Has Dropout Before Projection But No LayerNorm

**File**: model/heads.py lines 55-69

Standard practice (WeNet, ESPnet, NeMo) inserts a `LayerNorm` before the CTC projection linear layer to stabilize representations from different exit depths. Currently the CTC head is just `Dropout -> Linear -> log_softmax`. Adding a LayerNorm is especially important for multi-exit architectures where the representations at Layer 4, 8, and 12 have very different magnitudes.

---

### 2.5 MoE Additive Fusion Architecture: Asymmetric Residual Scales

**File**: moe/moe_layer.py lines 79-130

The current fusion is:
```
shared_out = self.combining_ffn(x)     # internally: residual + 0.5 * FFN(x)
expert_out = ...                        # weighted expert contribution (no residual)
out = shared_out + expert_out
```

`shared_out` already contains the full residual `x + 0.5 * FFN(x)`. Then adding `expert_out` (which in Phase 4B is `0.5 * expert_linear(x)` without residual) creates an asymmetry. The expert contribution has a fundamentally different scale than the shared trunk contribution.

---

### 2.6 Relative Positional Encoding is Missing

**File**: model/conformer.py lines 14-36

The model uses absolute sinusoidal positional encoding. Modern Conformer variants (WeNet, ESPnet, NeMo) all use **Relative Positional Encoding (RPE)** or **Rotary Positional Embeddings (RoPE)** for better generalization to variable-length sequences during streaming inference. For a streaming system with dynamic chunk sizes, RPE would significantly improve robustness.

---

## 3. Training Methodology Issues

### 3.1 CRITICAL: Stage 2 Final CER Is Very High (~25% Median, 84% at Step 10000)

**Evidence**: stage2_causal_metrics.csv shows:
- Mean sample CER over full training: 25.1%
- Final step (10000) sample CER: **84.0%**
- CTC Loss still at 2.78 (only dropped from 12.9 to 2.78)

The CER at step 10000 being 84% suggests the model was evaluated on a particularly hard sample (possibly long or dialectal), but the overall trajectory shows the CER oscillates wildly. This extreme variance indicates:

1. **Insufficient training duration**: 10,000 steps at batch_size=32 = ~320,000 utterances seen. For 1,450 hours of audio, this is less than 1 epoch of supervised data. Most ASR models train for 50-200 epochs.
2. **No curriculum on chunk sizes**: The uniform probability distribution `[0.25, 0.25, 0.20, 0.15, 0.15]` means 25% of batches use chunk_size=1 (strictly causal, 40ms). This is an extremely hard regime that produces near-random CER for a model still learning phonetics. Starting with full-context and gradually introducing causal constraints would produce smoother convergence.

---

### 3.2 CRITICAL: Stage 4 MoE Aux Load Balancing Loss Is Computed But Never Used

**File**: moe/train_moe.py lines 160-167

```python
# Collect load balance losses across all MoE layers
total_aux_loss = 0.0
for layer_idx in [4, 5, 6, 7, 8, 9, 10, 11]:
    block = model.encoder.layers[layer_idx]
    if isinstance(block.ffn2, SparseMoELayer):
        # Gated aux loss is tracked during block forward
        pass

total_loss = ctc_loss  # aux_loss is NEVER added!
```

The `SparseMoELayer` computes `self.current_aux_loss` during forward, but the training loop has a `pass` placeholder and never actually reads or adds it to the total loss. This means **no load balancing pressure exists** during Phase 4B.

Without load balancing, the router will collapse onto a single expert (the one that happened to receive slightly better gradients early on), causing the other two experts to become dead parameters. The fact that MoE still seems to work is likely because Phase 4A hard routing already differentiated the experts somewhat (though see issue 3.3), but the dynamic routing in 4B has no incentive to distribute load.

**Fix**:
```python
for layer_idx in [4, 5, 6, 7, 8, 9, 10, 11]:
    block = model.encoder.layers[layer_idx]
    if isinstance(block.ffn2, SparseMoELayer):
        total_aux_loss += block.ffn2.current_aux_loss
total_loss = ctc_loss + 0.01 * total_aux_loss
```

---

### 3.3 IMPORTANT: MoE Phase 4A Hard Routing Never Actually Activates

In Phase 4A (moe/train_moe.py line 140), `is_phase_4a` is checked, but `SparseMoELayer.forward()` is called via `model.forward_ctc()`, which doesn't pass `dialect_idx` or `use_hard_routing` at all. The `ConformerBlock.forward()` calls `self.ffn2(x)` with no extra arguments.

Looking at ConformerBlock.forward() (model/conformer.py lines 237-243):
```python
x = self.ffn2(x)
```

And SparseMoELayer.forward() signature:
```python
def forward(self, x, dialect_idx=None, use_hard_routing=False):
```

Since `dialect_idx` is never passed, **Phase 4A hard routing never actually happens**. The model always falls through to the Phase 4B dynamic routing path, even during steps 1-3,000. The "Phase 4A (Hard Routing)" log label is cosmetic only.

---

### 3.4 IMPORTANT: Training Duration Is Too Short Across All Stages

| Stage | Steps | Effective Data Seen | Industry Standard |
| :--- | :--- | :--- | :--- |
| Stage 1 (SSL) | 10,000 | ~1,450 hrs x 1 pass | 100k-400k steps (wav2vec 2.0, HuBERT) |
| Stage 2 (CTC) | 10,000 | ~320k utterances | 50-200 epochs over supervised data |
| Stage 3 (Multi-Exit) | 10,000 | ~240k utterances | 20-50k steps |
| Stage 4 (MoE) | 10,000 | ~240k utterances | 10-30k steps |

The SSL pretraining at 10k steps is **100x shorter** than standard self-supervised speech models. The learned representations are likely superficial — the Phase 3 gate passes (degradation ratio > 1.15) but just barely, suggesting the model learned local spectral structure but not deep acoustic-phonetic patterns.

---

### 3.5 No Gradient Accumulation Used Anywhere

With batch_size=24 and audio lengths of 3-12 seconds, the effective batch in audio-seconds is small (~100 audio-seconds per step). ASR training typically uses effective batch sizes of 500-2000 audio-seconds. Gradient accumulation over 4-8 steps would improve optimization stability significantly.

---

## 4. Data Pipeline Problems

### 4.1 Stages 2-3 Stream from HuggingFace Remote; Stage 4 Loads from Local RESPIN

Stages 2 and 3 use `StreamingCTCPrefetchLoader` which streams from HuggingFace (Shrutilipi + Vaani). Stage 4 uses `LocalRESPINPrefetchLoader` from local RESPIN data. This means:

1. **Different data distributions**: Stages 2-3 see Shrutilipi + Vaani (read speech + field recordings). Stage 4 sees only RESPIN (field recordings across dialects). The domain shift is significant.
2. **Network dependency**: Stages 2-3 are fragile to DNS/network failures (already encountered during Stage 3).
3. **No speed perturbation, pitch augmentation, or noise injection** in any data pipeline.

**Fix**: Download and cache all audio locally. Add speed perturbation (0.9x, 1.0x, 1.1x) and additive noise augmentation.

---

### 4.2 `get_combined_stream()` Is Sequential, Not Interleaved

**File**: pretraining/dataset_loader.py lines 107-117

```python
def get_combined_stream(...):
    if include_shrutilipi:
        yield from stream_shrutilipi(shrutilipi_dialects)
    if include_vaani:
        yield from stream_vaani(vaani_dialects)
```

This yields **all** Shrutilipi data first, then **all** Vaani data. The model sees one full corpus before the other, creating a distribution shift mid-training. True interleaving (round-robin or probabilistic sampling) would prevent catastrophic forgetting.

---

### 4.3 No Length Bucketing or Dynamic Batching

All data loaders use fixed batch sizes with random sampling. This means a batch might contain one 1-second clip and one 12-second clip, wasting 91% of compute on padding. Length-sorted bucketing would improve GPU utilization by 30-50%.

---

## 5. Metric & Evaluation Integrity Issues

### 5.1 CRITICAL: Stage 4 CER Is Measured Without CTC Decoding (Uses `decode()` Not `ctc_decode()`)

**File**: moe/train_moe.py lines 185-186

```python
pred_tokens = logits_l12[0].argmax(dim=-1).tolist()
pred_text = tokenizer.decode(pred_tokens)
```

This calls `tokenizer.decode()` which does NOT perform CTC blank collapsing or repetition removal. The logged CER of ~40% in Stage 4 is **massively inflated** because the "predicted text" contains every repeated character and blank tokens. The actual post-CTC-decode CER is likely 15-25% lower.

Compare to Stage 2 (train_causal.py line 199) which correctly uses `tokenizer.ctc_decode(first_pred_ids)`.

**This same bug exists in** calibration/train_calibration.py line 178.

---

### 5.2 CER Is Computed On a Single Sample Per Batch, Not Batch Average

All training loops (Stages 2, 3, 4, 5) compute CER on `batch["texts"][0]` — just the first sample. This is a high-variance point estimate. A single easy or hard sample can swing the reported CER by 50 percentage points. The logged CER curves have extreme oscillation because of this.

**Fix**: Compute CER over all samples in the batch (or at least 4-8) and report the mean.

---

### 5.3 `compute_cer()` Has O(n^2) Space Complexity

**File**: causal_alignment/train_causal.py lines 40-58

The CER implementation uses a full `(len(r)+1) x (len(h)+1)` matrix. For long utterances (200+ characters), this allocates 40,000+ cells every step. Using the standard two-row optimization would reduce memory from O(n^2) to O(n). Alternatively, use the `jiwer` library which is already installed.

---

### 5.4 Eval Benchmark RTF Calculation Is Approximate

**File**: eval_final_benchmark.py

RTF for Layer 4 and Layer 8 exits is computed as `batch_infer_time * (layer / 12.0)`. This linearly scales the **full model's** inference time, but early exits actually save compute by not executing later layers at all. The correct approach is to run inference separately for each exit depth with `torch.cuda.synchronize()` timing.

---

## 6. Code Quality & Engineering Debt

### 6.1 `dataset.py` References `time.sleep()` Without Importing `time`

**File**: causal_alignment/dataset.py line 141

Line 141: `time.sleep(5)` — but `time` is never imported. This will crash on network errors.

---

### 6.2 Hardcoded Paths and Magic Numbers Throughout

- `respin_split_cache.json` path is hardcoded as `"moe/respin_split_cache.json"` in multiple files.
- Model architecture parameters (`d_model=256`, `num_layers=12`, `vocab_size=105`) are duplicated in 8+ locations instead of using a config object.
- Conversation-specific task log paths are hardcoded in telemetry/archive_all_metrics.py.

---

### 6.3 No Unit Tests Exist

There are zero test files. The codebase has no automated verification that:
- Model forward/backward passes produce valid gradients.
- Tokenizer encode/decode roundtrips correctly.
- CTC loss accepts the tensor shapes produced by the model.
- Checkpoint save/load preserves weights exactly.

---

### 6.4 `pyproject.toml` Says `requires-python = ">=3.13"` But .python-version May Differ

Python 3.13 is very new. The codebase uses `uv` which may install 3.11 or 3.12. This should be aligned.

---

## 7. Prioritized Improvement Roadmap

### Tier 1: Fix Before Stage 5 Finishes (Estimated Impact: +5-10% CER improvement)

| # | Issue | File(s) | Status | Effort |
| :--- | :--- | :--- | :--- | :--- |
| **T1.1** | Remove double `log_softmax` in train_moe.py and train_calibration.py | moe/train_moe.py, calibration/train_calibration.py | ✅ Completed & Verified | 5 min |
| **T1.2** | Fix `decode()` to `ctc_decode()` in train_moe.py and train_calibration.py | Same files | ✅ Completed & Verified | 5 min |
| **T1.3** | Add load balancing loss to Stage 4 total_loss | moe/train_moe.py | ✅ Completed & Verified | 10 min |
| **T1.4** | Add SpecAugment to `forward_ctc()` | model/asr_model.py | ✅ Completed & Verified | 5 min |
| **T1.5** | Fix MoE Phase 4B expert residual connection | moe/moe_layer.py | ✅ Completed & Verified | 15 min |

### Tier 2: Fix Before Final Evaluation (Estimated Impact: +3-5% CER, correct metrics)

| # | Issue | File(s) | Status | Effort |
| :--- | :--- | :--- | :--- | :--- |
| **T2.1** | Fix GRPO router collapse (add entropy bonus, anneal latency) | multi_exit/grpo_loss.py, multi_exit/train_multiexit.py | ✅ Completed & Verified | 1 hr |
| **T2.2** | Pass `dialect_idx` through ConformerBlock to MoE layer | model/conformer.py, moe/train_moe.py | ✅ Completed & Verified | 30 min |
| **T2.3** | Batch-average CER logging instead of single-sample | All training scripts | ✅ Completed & Verified | 30 min |
| **T2.4** | Add `import time` to `causal_alignment/dataset.py` | causal_alignment/dataset.py | ✅ Completed & Verified | 1 min |
| **T2.5** | Add LayerNorm to CTCHead | model/heads.py | ✅ Completed & Verified | 10 min |

### Tier 3: Structural Improvements for Next Training Run

| # | Issue | File(s) | Status | Effort |
| :--- | :--- | :--- | :--- | :--- |
| **T3.1** | Implement Relative Positional Encoding (RoPE) | model/conformer.py | ✅ Completed & Verified | 3 hrs |
| **T3.2** | Add speed perturbation + noise augmentation | moe/respin_dataset.py, causal_alignment/dataset.py | ✅ Completed & Verified | 2 hrs |
| **T3.3** | Implement dynamic batching / length bucketing | moe/respin_dataset.py | ✅ Completed & Verified | 3 hrs |
| **T3.4** | Add gradient accumulation (4-8 steps) | All training scripts | ✅ Completed & Verified | 1 hr |
| **T3.5** | Interleave data streams instead of sequential | pretraining/dataset_loader.py | ✅ Completed & Verified | 1 hr |
| **T3.6** | Centralize model config (YAML/dataclass) | config.py | ✅ Completed & Verified | 2 hrs |
| **T3.7** | Train for configurable steps across all stages | All training scripts | ✅ Configured | Compute time |
| **T3.8** | Implement checkpoint retention & validation tracking | telemetry/metrics_logger.py | ✅ Completed & Verified | 2 hrs |
| **T3.9** | Correct RTF measurement with per-exit inference | eval_final_benchmark.py | ✅ Completed & Verified | 1 hr |


---

### Summary of Expected Impact

**Fixing just Tier 1 items (T1.1-T1.5) before Stage 5 completes would likely improve the final benchmark CER by 5-10 percentage points absolute.** The double log_softmax alone has been silently degrading gradient quality across 17,000+ training steps (all of Stage 4 + Stage 5). The missing SpecAugment has allowed unchecked overfitting. The broken expert residual connection has been hamstringing the MoE expert capacity.

**The MoE load balancing loss was computed but never backpropagated through the graph.** This means the 3 dialect experts in Phase 4B had no explicit pressure to distribute load, potentially causing expert collapse where 1-2 experts dominate and the third becomes a dead subnetwork. This is the single most impactful training bug for the dialect specialization objective.
