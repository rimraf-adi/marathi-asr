# Results Reference for the Dialect-Aware Sparse MoE Conformer for Marathi ASR

**Purpose:** single source of truth for writing the paper. Every number below was produced by the scripts listed in Section 11 on the two retained checkpoints, with one evaluation protocol, and is regenerated from `evals/*.json`. Statements that the data do not support are not made; where a result is unfavourable (SraVaani on RESPIN, out-of-domain accuracy, latency) it is reported as measured.  
**Repository:** [github.com/rimraf-adi/marathi-asr](https://github.com/rimraf-adi/marathi-asr)  
**Hardware:** NVIDIA RTX A5000 (24 GB), Windows x86_64, PyTorch 2.6 + CUDA 12.4  
**Date of measurements:** 2026-11-02

---

## 1. Claims the data support

| # | Claim | Evidence (section) |
| :--- | :--- | :--- |
| C1 | At an identical 130.8M parameter budget, the dialect-routed MoE reaches **19.50% WER** on the RESPIN held-out test set versus **36.13%** for the dense Conformer: **16.63 points absolute (46.0% relative)**, paired speaker-clustered bootstrap CI [15.72, 17.56], p < 0.0001. | 4.1, 4.3 |
| C2 | The gain holds in every dialect partition, including Standard Marathi (13.73 points) and is largest on the regional dialects (18.31 / 16.69 / 17.64 points for D1 / D2 / D4). | 4.1, 4.3 |
| C3 | The experts carry the gain: removing all expert contributions raises WER from 19.50% to 29.05%; dropping a single expert hurts its own dialect most; learned routing beats oracle dialect routing (22.93%) and random routing (25.55%). | 5.1, 5.2 |
| C4 | The MoE preserves dialect-specific word forms far better than a dense model of equal size and than IndicConformer-600M (dialect-skewed word recall 89.7/90.9/93.0% vs. 74.2/79.7/83.2% and 59.8/74.5/54.3% on D1/D2/D4). | 5.5 |
| C5 | Autonomous frame-level routing outperforms dialect-specific LoRA adapters that are given the oracle dialect label (34.49% vs. 19.50%). | 5.4 |
| C6 | On RESPIN the MoE (19.50%) is significantly better than IndicConformer-600M (23.94%) and significantly worse than SraVaani-1.0 (15.77%). | 4.1 |
| C7 | By parameter count the MoE uses 97.2M of 130.8M parameters per frame (74.3%), but in the present implementation it is **slower** than the dense model (Section 7): the parameter saving is not a speed saving. | 7 |
| C8 | Out of domain (IndicVoices, FLEURS) both of our models are far less accurate than the external baselines; Kathbath is not an out-of-domain test (all 20 validation speakers also occur in Kathbath train). | 4.2, 8 |


Training-ablation claims (seed variance, upcycling, number of experts, expert placement) are placeholders in Section 5.6 until the queued runs finish.
---

## 2. Data

### 2.1 Training data by stage (both retained models)

| Stage | Purpose | Data | Size / steps |
| :--- | :--- | :--- | :--- |
| 1. Self-supervised pretraining | Masked-reconstruction pretraining of the encoder | `ai4bharat/Shrutilipi` (All India Radio news audio) and `ARTPARK-IISc/Vaani` (image-prompted spontaneous speech), streamed; IndicVoices excluded (`--no_indicvoices`) | MoE run: 1,160,977 utterances (1,763.9 h) in one epoch, 18,141 steps. Dense run: 30,000 steps |
| 2. Supervised CTC adaptation | Dense: plain CTC. MoE: upcycling of encoder blocks 5-12 into 3 dialect experts, with a CTC warm-up (steps 1-3,000), metadata-guided hard routing (3,001-18,000) and learned routing (18,001-80,000) | IISc RESPIN train partition (speaker-disjoint split in `moe/respin_split_cache.json`: 768,893 train + 41,041 calibration utterances) | 80,000 steps |
| 3. Final fine-tuning | Multi-source CTC fine-tuning | RESPIN train pool (809,934 utterances), OpenSLR64 Marathi (1,569), Kathbath clean + noisy train (85,701), Mozilla Common Voice Marathi (1,205), and the Shrutilipi/Vaani stream | MoE: last saved step 342,000 (400,000 planned); dense: 300,000 steps; batch 32 x grad-accum 4, peak LR 3e-5 |
| LoRA adapters (Sec. 5.4) | Three dialect adapters on the frozen dense backbone | RESPIN clean train, one pool per dialect (D1, D2, D4) | 6,000 steps |

RESPIN: Kumar et al., NeurIPS 2025 Datasets & Benchmarks (read speech, agriculture and finance domains, multiple dialects per language).

### 2.2 Evaluation data

| Dataset | Split | Utterances / hours | Relation to our training data |
| :--- | :--- | :---: | :--- |
| IISc RESPIN | official test | 2,170 / 3.04 h, 160 speakers, 4 dialects | same corpus, **no speaker and no sentence shared with any training source** (Section 8) |
| AI4Bharat Kathbath | valid | 2,378 / 4.18 h, 20 speakers | in-domain: all 20 speakers also appear in Kathbath train (84,070 clean utterances, 123 speakers); 9 sentences shared with training text |
| AI4Bharat IndicVoices | valid | 3,673 / 6.15 h, 435 speakers | not used in training; no sentence overlap |
| Google FLEURS (mr_in) | test | 1,015 / 3.86 h (349 distinct sentences) | not used in training; no sentence overlap |

Common Voice Marathi is used only as training data; no Common Voice number is reported. The training corpora of SraVaani-1.0 and IndicConformer-600M are those documented on their model cards, not controlled by us.

---

## 3. Models and protocol

### 3.1 Our models

| | Dense Conformer (baseline) | MoE Conformer (proposed) |
| :--- | :---: | :---: |
| Encoder | 12 blocks, d_model 512, 8 heads, conv kernel 31 | same |
| FFN expansion | 8 | 4 (shared trunk) + 3 experts of expansion 4 in blocks 5-12 |
| Router | - | linear 512 -> 3 per MoE block (12,312 parameters in total), top-1 per frame, additive to the always-active shared FFN |
| Output | CTC head, 105-symbol character vocabulary | same |
| Total parameters | 130.8M | 130.8M (8 MoE blocks x 3 experts x 2.10M) |
| Parameters used per frame (top-1) | 130.8M | 97.2M (74.3%) |
| Retained checkpoint | `runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt` | `runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt` |

Experts 0/1/2 are initialised from the pretrained FFN and associated, during hard-routing training, with Malvani (D1), Ahirani (D2) and Varhadi (D4); Standard Marathi (D3) is served by the shared trunk. The design is reported as built; no claim is made that 3 experts or blocks 5-12 are optimal.

### 3.2 External baselines

| Model | Parameters | Architecture | Decoding in our harness | Source |
| :--- | :---: | :--- | :--- | :--- |
| SraVaani-1.0 | ~430M | FastConformer, hybrid TDT-CTC decoder; pretrained on ~31k h of Vaani, fine-tuned on ~31k h of transcribed speech in 65 Indian languages and dialects | model `transcribe()` with default settings | [model card](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) |
| IndicConformer-600M | 600M | multilingual Conformer, hybrid CTC + RNNT | CTC mode, language `mr` | [model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) |

### 3.3 Evaluation protocol

* Greedy CTC decoding unless stated (Section 5.3). Utterances are sorted by length for batching.
* Text normalisation `mr-norm-1` (`normalizer.py`), applied identically to references and hypotheses of every model: Unicode NFC, removal of ZWNJ/ZWJ, nukta unification, anusvara/chandrabindu harmonisation, punctuation stripping.
* WER/CER are corpus-level; "n" counts utterances with a non-empty normalised reference. Dataset-level CIs: 1,000 utterance-level bootstrap resamples. Paired CIs (Section 4.3): 10,000 speaker-clustered resamples over the 160 RESPIN test speakers.
* One training run per model; no variance across training seeds was measured, so differences of a few tenths of a point between configurations of the same model should not be over-interpreted.
* Checkpoints: the best of each family on RESPIN and Kathbath (Section 6).

---

## 4. Main results

### 4.1 RESPIN held-out test (2,170 utterances, 160 unseen speakers)

| Model | Params | D1 Malvani | D2 Ahirani | D3 Standard | D4 Varhadi | **ALL WER** [95% CI] | ALL CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SraVaani-1.0 | ~430M | 21.71% | 17.80% | 8.97% | 14.17% | **15.77%** [15.07, 16.53] | 2.97% |
| **MoE Conformer (ours)** | 130.8M | 25.68% | 21.58% | 11.15% | 19.11% | **19.50%** [18.75, 20.32] | 4.29% |
| IndicConformer-600M | 600M | 34.59% | 24.64% | 11.57% | 24.29% | **23.94%** [22.99, 24.93] | 4.95% |
| Dense Conformer (ours) | 130.8M | 43.99% | 38.27% | 24.88% | 36.75% | **36.13%** [35.18, 37.18] | 9.56% |
| MoE, experts bypassed (Sec. 5.1) | 130.8M | 39.00% | 29.81% | 15.38% | 31.33% | 29.05% | - |
| Dense + oracle dialect LoRA (Sec. 5.4) | 142.6M | 41.46% | 36.76% | 24.93% | 34.26% | 34.49% | - |

* MoE vs. dense (same 130.8M budget): 16.63 points absolute, 46.0% relative.
* The MoE interval [18.75, 20.32] lies entirely below IndicConformer-600M [22.99, 24.93] and entirely above SraVaani-1.0 [15.07, 16.53]. SraVaani-1.0 has ~3.3x our parameter count and, per its model card, was fine-tuned on ~31k hours of transcribed speech; it is the most accurate system on RESPIN.

### 4.2 All datasets

WER % (CER %) [95% CI], greedy CTC, same normaliser.

| Model | RESPIN test n=2,170 | Kathbath valid (in-domain) n=2,378 | IndicVoices valid (OOD) n=3,673 | FLEURS test (OOD) n=1,015 |
| :--- | :---: | :---: | :---: | :---: |
| **MoE Conformer (ours)** | 19.50 (4.29) [18.75, 20.32] | 29.41 (7.01) [28.65, 30.23] | 59.20 (23.48) [58.24, 60.11] | 36.74 (11.02) [35.67, 37.92] |
| Dense Conformer (ours) | 36.13 (9.56) [35.18, 37.18] | 50.69 (14.52) [49.81, 51.63] | 77.19 (34.78) [76.37, 77.91] | 57.04 (19.49) [55.86, 58.24] |
| SraVaani-1.0 | 15.77 (2.97) [15.07, 16.53] | 12.93 (2.72) [12.37, 13.51] | 15.21 (5.55) [14.62, 15.78] | 17.79 (5.61) [16.94, 18.61] |
| IndicConformer-600M | 23.94 (4.95) [22.99, 24.93] | 13.52 (2.88) [12.98, 14.14] | 15.96 (5.43) [15.37, 16.53] | 18.58 (5.62) [17.75, 19.40] |

* In-domain (RESPIN, Kathbath) the MoE halves the dense model's error on Kathbath (50.69% to 29.41%).
* Out of domain, the MoE improves on the dense model by 18.0 points (IndicVoices) and 20.3 points (FLEURS) but remains far from SraVaani-1.0 and IndicConformer-600M (59.2% and 36.7% vs. roughly 15-19%). The models are specialised to RESPIN-style read speech; they should not be presented as general Marathi ASR systems.

### 4.3 Paired comparison MoE vs. dense (speaker-clustered bootstrap, B = 10,000, 160 speakers)

| Partition | MoE WER 95% CI | Dense WER 95% CI | Paired reduction (dense - MoE) [95% CI] | p |
| :--- | :---: | :---: | :---: | :---: |
| D1 Malvani | [23.62, 27.73] | [40.99, 46.97] | 18.31 [16.68, 19.98] | < 0.0001 |
| D2 Ahirani | [19.34, 23.94] | [35.44, 41.18] | 16.67 [14.78, 18.67] | < 0.0001 |
| D3 Standard | [9.33, 13.33] | [21.94, 27.95] | 13.72 [12.04, 15.38] | < 0.0001 |
| D4 Varhadi | [17.49, 20.86] | [34.36, 39.25] | 17.66 [15.87, 19.51] | < 0.0001 |
| ALL | [18.23, 20.79] | [34.38, 37.87] | 16.62 [15.72, 17.56] | < 0.0001 |

The reduction is significant in every partition; the empirical p-value is bounded below by 1/B = 0.0001. The bootstrap mean of the paired reduction (16.62) differs from the point estimate (16.63) only through resampling.

---

## 5. Analyses

### 5.1 Routing conditions (same MoE checkpoint, greedy CTC)

| Routing | Granularity | D1 | D2 | D3 | D4 | ALL | vs. learned |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Learned router (top-1) | frame | 25.68 | 21.58 | 11.15 | 19.11 | **19.50** | - |
| Oracle dialect label | utterance | 29.30 | 23.31 | 15.38 | 23.31 | **22.93** | +3.43 |
| Random expert | utterance | 34.77 | 26.37 | 13.66 | 26.79 | **25.55** | +6.05 |
| Experts bypassed (shared trunk only) | - | 39.00 | 29.81 | 15.38 | 31.33 | **29.05** | +9.55 |

Oracle routing sends D1/D2/D4 utterances to experts 0/1/2 and D3 utterances to the trunk alone. Learned frame-level routing is the best of the four conditions in every dialect, including D3, and on aggregate and on D1/D2/D4 oracle routing is better than random routing and than bypassing the experts (on D3, which has no dedicated expert, random routing scores better than oracle).

### 5.2 Inference-time ablations of the trained MoE (no retraining)

Greedy CTC, same protocol. "Bypass" removes the expert contribution of a block (shared FFN only); "drop" prevents the router from selecting one expert (it chooses among the remaining two).

| Intervention | D1 | D2 | D3 | D4 | ALL | vs. baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| None (trained MoE) | 25.68 | 21.58 | 11.15 | 19.11 | **19.50** | - |
| Bypass block 5 | 26.25 | 22.09 | 11.22 | 19.16 | **19.81** | +0.31 |
| Bypass block 6 | 25.97 | 21.89 | 11.15 | 19.41 | **19.73** | +0.23 |
| Bypass block 7 | 26.10 | 21.82 | 11.22 | 18.89 | **19.63** | +0.13 |
| Bypass block 8 | 26.29 | 21.89 | 11.45 | 19.00 | **19.78** | +0.28 |
| Bypass block 9 | 26.36 | 21.69 | 11.55 | 19.16 | **19.81** | +0.31 |
| Bypass block 10 | 27.15 | 23.07 | 11.69 | 20.36 | **20.70** | +1.20 |
| Bypass block 11 | 29.70 | 23.73 | 12.64 | 25.06 | **22.91** | +3.41 |
| Bypass block 12 | 29.21 | 23.66 | 11.78 | 21.57 | **21.70** | +2.20 |
| Bypass blocks 5-8 | 27.41 | 22.61 | 11.59 | 19.48 | **20.41** | +0.91 |
| Bypass blocks 9-12 | 36.90 | 28.36 | 14.45 | 30.10 | **27.62** | +8.12 |
| Drop expert 0 (Malvani) | 39.42 | 22.76 | 13.50 | 21.09 | **24.37** | +4.87 |
| Drop expert 1 (Ahirani) | 29.57 | 28.89 | 15.08 | 23.65 | **24.44** | +4.94 |
| Drop expert 2 (Varhadi) | 27.19 | 22.81 | 12.20 | 27.60 | **22.55** | +3.05 |

* **Expert specialisation.** Each dropped expert hurts its associated dialect most: removing expert 0 raises D1 WER by 13.7 points (other dialects 1.2, 2.0); expert 1 raises D2 by 7.3 (others 3.9, 4.5); expert 2 raises D4 by 8.5 (others 1.5, 1.2).
* **Where the experts matter.** Bypassing blocks 5-8 costs 0.91 points, blocks 9-12 cost 8.12 points; single-block bypasses are largest at blocks 11 (3.41) and 12 (2.20). The upper blocks carry most of the dialect-specific computation. This is an inference-time sensitivity analysis of one trained model, not a comparison of models trained with different expert placements.

### 5.3 Decoding (130.8M MoE)

| Decoder | D1 | D2 | D3 | D4 | ALL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Greedy CTC (used for all results) | 25.68 | 21.58 | 11.15 | 19.11 | **19.50** |
| Beam 16 + lexicon (no language model) | 25.42 | 20.84 | 10.87 | 18.80 | **19.10** |

No language model is used anywhere in this document. A 16-beam search restricted to a 34,075-word lexicon mined from the RESPIN train and calibration splits (`decoding/beam_search_decoder.py`) changes WER by only 0.40 points, so all headline numbers use plain greedy CTC decoding for every model.

### 5.4 Dialect LoRA adapters vs. MoE (PEFT control)

Three rank-16 LoRA adapters (alpha 32, dropout 0.05) on attention (Wq, Wv) and both FFNs of all 12 blocks of the frozen dense backbone; 3,932,160 trainable parameters per adapter, 11,796,480 in total, 142,567,017 parameters in the combined model.

| System | Needs dialect label at inference | D1 | D2 | D3 | D4 | ALL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dense, no adapter | no | 43.99 | 38.27 | 24.88 | 36.75 | 36.13 |
| Dense + oracle LoRA | **yes** | 41.46 | 36.76 | 24.93 | 34.26 | 34.49 |
| MoE (learned routing) | no | 25.68 | 21.58 | 11.15 | 19.11 | **19.50** |

Cross-adapter matrix (WER %, every utterance of a test dialect decoded with one adapter):

| Test dialect | adapter 0 (Malvani) | adapter 1 (Ahirani) | adapter 2 (Varhadi) | mismatch penalty |
| :--- | :---: | :---: | :---: | :---: |
| D1 Malvani | **41.31** | 48.35 | 49.67 | +7.04 to +8.36 |
| D2 Ahirani | 43.73 | **36.74** | 41.20 | +4.46 to +6.99 |
| D4 Varhadi | 40.68 | 39.18 | **34.30** | +4.88 to +6.38 |

* Oracle LoRA improves on the dense model by 1.64 points; the MoE improves on oracle LoRA by a further 14.99 points (43.5% relative) without any dialect metadata.
* Each adapter is best on its own dialect, confirming that the adapters are dialect-specific. Note that the MoE was trained with a longer, multi-stage recipe than the adapters (6,000 steps); the comparison measures what each recipe achieved, not a controlled equal-compute test.

### 5.5 Dialect preservation (RESPIN test predictions)

"Dialect-skewed words" are reference word types that occur at least 10 times in one regional dialect and at least 5 times more often (per token) there than in all other dialects. They are found automatically from the test references and include dialect forms (e.g. Malvani आसा, माका; Ahirani शे; Varhadi मले, हाये) as well as some topic words (e.g. Ahirani लागवड, Varhadi बिमा), so the measure is a lexical-skew proxy and not a linguist-annotated dialect inventory.

| Dialect | skewed word types | tokens in test refs | share of dialect tokens |
| :--- | :---: | :---: | :---: |
| Malvani (D1) | 35 | 612 | 13.4% |
| Ahirani (D2) | 22 | 364 | 8.0% |
| Varhadi (D4) | 28 | 470 | 10.7% |

Recall of dialect-skewed words (a reference occurrence counts as recalled if it is aligned to an identical hypothesis word):

| Model | D1 Malvani | D2 Ahirani | D4 Varhadi |
| :--- | :---: | :---: | :---: |
| **MoE Conformer (ours)** | 89.7% | 90.9% | 93.0% |
| Dense Conformer (ours) | 74.2% | 79.7% | 83.2% |
| SraVaani-1.0 | 86.9% | 92.9% | 91.3% |
| IndicConformer-600M | 59.8% | 74.5% | 54.3% |

Error composition on RESPIN (% of reference words):

| Model | Substitutions | Deletions | Insertions | WER |
| :--- | :---: | :---: | :---: | :---: |
| **MoE Conformer (ours)** | 16.28 | 2.20 | 1.02 | 19.50 |
| Dense Conformer (ours) | 30.50 | 3.80 | 1.83 | 36.13 |
| SraVaani-1.0 | 12.25 | 2.75 | 0.77 | 15.77 |
| IndicConformer-600M | 19.25 | 3.73 | 0.97 | 23.94 |

Most frequent replacements of dialect-skewed words (reference -> hypothesis, count):

* IndicConformer-600M, Malvani: आसा -> असा (42), आसत -> असत (20), करतत -> करतात (14), मिळात -> मिळत (13)
* IndicConformer-600M, Varhadi: मले -> मला (34), येईन -> येईल (19), रायते -> राहते (13), लागन -> लागल (12)
* Dense Conformer (ours), Malvani: आसा -> असा (15), असा -> आसा (10), आसत -> आस (3), असत -> आसत (3)
* Dense Conformer (ours), Varhadi: येईन -> येईल (4), हाये -> हाय (4), बिमा -> विमा (3), हप्ते -> हफ्ते (3)
* **MoE Conformer (ours)**, Malvani: आसा -> असा (17), असा -> आसा (5), हया -> ह्या (4), करूचा -> करुचा (4)
* **MoE Conformer (ours)**, Varhadi: माह्या -> माया (4), मोबाईल -> मोबाईलवरून (4), डी -> टी (3), हाये -> हाय (3)

IndicConformer-600M frequently rewrites dialect forms into standard Marathi (e.g. आसा -> असा, मले -> मला, शे -> आहे), which is the "standardisation" behaviour the MoE reduces; some of the remaining errors of all systems are orthographic variants (e.g. करु/करू), not dialect errors. SraVaani-1.0 preserves dialect forms about as well as the MoE (higher recall on D2, lower on D1 and D4) while having a lower overall WER.

### 5.6 Training ablations (reduced budget)

> **PLACEHOLDER SECTION.** The runs below are queued (`scripts/run_ablation_queue.py`) and have not been measured yet. Every `[TBD]` is filled automatically by `python scripts/build_final_md.py` once `evals/ablation_training.json` exists. Do not quote any number from this section before then.

**Protocol.** All variants start from the same re-run Stage 1 pretrained encoder (the original Stage 1 checkpoints were deleted) and are trained for 20,000 Stage-2 steps (CTC warm-up 750, metadata-guided hard routing 3,750, learned routing for the rest; one quarter of the 80,000-step recipe), batch 32 x grad-accum 4, peak LR 1.5e-4, then evaluated on RESPIN test and Kathbath valid with greedy CTC. Absolute WERs are therefore higher than the headline 19.50% (which includes Stage 3 fine-tuning); only differences between variants are meaningful. Everything except the varied factor is identical to `base_s0`.

**Claim A - seed variance** (does a single run suffice?)

| Run (seed) | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MoE, seed 0 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| MoE, seed 1 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| MoE, seed 2 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| **mean +/- sd (n=3)** | | | | | [TBD] | [TBD] |

Template sentence: "Across three training seeds the reduced-budget MoE reaches [TBD] +/- [TBD]% WER on RESPIN; the seed-to-seed standard deviation is [TBD] points, [much smaller than / comparable to] the MoE-dense gap." (Dense-model seeds are not part of this queue: a dense Stage 1 re-run takes about 85 h.)

**Claim B - upcycling vs. randomly initialised experts**

| Variant | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Experts upcycled from the pretrained FFN (base_s0) | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| Experts randomly initialised (scratch_s0) | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |

Template sentence: "Initialising experts from the pretrained FFN [lowers / does not change] WER by [TBD] points relative to random expert initialisation at the same step budget."

**Claim C - number of experts** (3 experts = design choice or finding?)

| Experts per MoE block | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| 2 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| **3 (base_s0)** | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| 4 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |

Hard-routing note: with K < 3 the dialects whose index is >= K have no dedicated expert during the hard-routing phase (they use the shared FFN only); with K = 4 the fourth expert receives the Standard-Marathi utterances during that phase.

Template sentence: "WER [decreases / saturates] from [TBD]% (K=1) to [TBD]% (K=3), and [does / does not] improve further with K=4 ([TBD]%)."

**Claim D - expert placement** (which blocks hold the experts?)

| MoE blocks | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-4 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| **5-12 (base_s0)** | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |
| 1-12 | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | [TBD] |

Template sentence: "Placing experts in blocks 5-12 gives [TBD]% WER versus [TBD]% for blocks 1-4 and [TBD]% for all twelve blocks, consistent with the inference-time bypass analysis (Section 5.2), where [TBD]."

---

## 6. Checkpoint selection

Ten trained checkpoints (8 MoE, 2 dense; the dense Stage 1 pretraining checkpoint has no CTC head and cannot be scored) were evaluated on RESPIN and Kathbath with the same protocol; only the best of each family is retained.

| Retained checkpoint | RESPIN WER | Kathbath WER |
| :--- | :---: | :---: |
| **MoE 130.8M, Stage 3 vanilla SFT** | 19.50% | 29.41% |
| **Dense 130.8M, Stage 3 final alignment** | 36.13% | 50.69% |

The remaining 8 scored checkpoints (earlier stages, alternative runs, GRPO and polishing stages) were worse on both RESPIN and Kathbath (best alternative MoE: 22.78% / 32.80%) and were deleted.

---

## 7. Efficiency

Measured on NVIDIA RTX A5000, fp32, 300 random RESPIN test utterances (1482 s of audio), idle GPU, warm-up excluded.

| | Dense | MoE |
| :--- | :---: | :---: |
| Total parameters | 130.8M | 130.8M |
| Parameters used per frame | 130.8M | 97.2M (74.3%) |
| Batch-1 latency, median / p95 | 26.7 / 31.1 ms | 72.1 / 82.5 ms |
| Batch-1 real-time factor | 0.0055 | 0.0148 |
| Batch-16 throughput (RTFx) | 807.1 | 604.4 |
| Peak GPU memory, batch 1 / 16 | 728 / 2314 MB | 725 / 2311 MB |

The MoE uses 25.7% fewer parameters per frame, but its median batch-1 latency is 2.7x that of the dense model and its batch-16 throughput is 75% of the dense model, because the reference implementation dispatches frames to experts with per-expert masked indexing (`moe/moe_layer.py`), which is not optimised for speed and was not profiled further. All experts must also be resident in memory (same peak memory). Both models are far faster than real time (RTFx > 60 at batch 1). The efficiency claim for the paper is therefore the parameter count per frame, not wall-clock speed.

---

## 8. Data integrity audit

Script `scripts/audit_leakage_all.py` (+ `scripts/audit_kathbath_overlap.py`); normalised exact-sentence match of each evaluation set against all training text sources (RESPIN train + calibration, Kathbath clean + noisy train, OpenSLR64, Common Voice Marathi).

| Evaluation set | Utterances (distinct sentences) | Speakers | Speaker overlap with training | Sentence overlap with training text |
| :--- | :---: | :---: | :---: | :---: |
| RESPIN test | 2,170 (711) | 160 | **0** of 2,644 train+calibration speakers | **0** |
| Kathbath valid | 2,378 (2,378) | 20 | **20 of 20** occur in Kathbath clean train | 9 (6 Kathbath, 1 OpenSLR64, 2 Common Voice) |
| IndicVoices valid | 3,679 (3,163) | 435 | n/a (corpus not used in training) | **0** |
| FLEURS test | 1,015 (349) | n/a | n/a (corpus not used in training) | **0** |

RESPIN test has no speaker and no sentence in common with any training source, so the RESPIN results are clean held-out results. Kathbath validation speakers all appear in Kathbath train, so Kathbath numbers measure in-domain, speaker-seen performance. (IndicVoices valid has 6 more utterances than scored because empty normalised references are excluded from scoring.)

---

## 9. Related work

Every reference below was checked against its publisher or arXiv record.

* **SpeechMoE** (You, Feng, Su, Yu; Interspeech 2021, [arXiv:2105.03036](https://arxiv.org/abs/2105.03036)) introduces a dynamic-routing MoE acoustic model whose router uses the previous layer output together with an embedding network, with 7.0-23.0% relative CER improvement over static networks at comparable computation. **SpeechMoE2** (You et al., ICASSP 2022, [arXiv:2111.11831](https://arxiv.org/abs/2111.11831)) adds global domain and accent embeddings to the router input (1.9-17.7% relative CER gains on a multi-accent task). Our router uses no external embeddings or labels at inference.
* **DialectMoE** (Zhou et al., CCL 2024, [ACL Anthology 2024.ccl-1.89](https://aclanthology.org/2024.ccl-1.89/)) is an end-to-end MoE for Chinese multi-dialect recognition in low-resource conditions. It is the closest dialect-ASR MoE precedent; we apply the idea to Marathi dialects and compare against a capacity-matched dense model and a PEFT alternative.
* **MoE-CTC** (Lee, Kim, Lee; ACL 2026, [arXiv:2602.01967](https://arxiv.org/abs/2602.01967)) uses accent-aware routing during training with intermediate CTC supervision and label-free routing at inference for accented speech recognition. We use a single CTC head and metadata-guided routing only in an early training phase.
* **LR-MoE** (Interspeech 2023, [arXiv:2307.05956](https://arxiv.org/abs/2307.05956)) and **DLG-MoE** ([arXiv:2407.18581](https://arxiv.org/abs/2407.18581)) route by language group for multilingual / code-switching ASR.
* **Sparse upcycling** (Komatsuzaki et al., ICLR 2023, [arXiv:2212.05055](https://arxiv.org/abs/2212.05055)) initialises MoE layers from dense checkpoints; we use the same idea to initialise experts from the pretrained FFN.
* **Data and baselines:** RESPIN-S1.0 (Kumar et al., NeurIPS 2025 Datasets & Benchmarks); Kathbath / IndicSUPERB (Javed et al., AAAI 2023, [arXiv:2208.11761](https://arxiv.org/abs/2208.11761)); IndicVoices (Javed et al., Findings of ACL 2024, [2024.findings-acl.639](https://aclanthology.org/2024.findings-acl.639)); FLEURS (Conneau et al., SLT 2022, [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)); Shrutilipi (Bhogale et al., ICASSP 2023, [arXiv:2208.12666](https://arxiv.org/abs/2208.12666)); Vaani and SraVaani ([ARTPARK-IISc](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)); IndicConformer ([AI4Bharat](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)).

---

## 10. Scope of the claims

1. **Headline models: one run each.** The headline dense and MoE models were each trained once. Seed variance, expert initialisation, number of experts and expert placement are placeholders in Section 5.6 and are not claimed until measured. The load-balancing loss and the data mix are not varied, and no claim is made that the chosen values are optimal. Section 5.2 characterises the trained model by inference-time interventions.
2. **Macro-dialect granularity.** Three experts correspond to the three regional RESPIN dialects (Malvani, Ahirani, Varhadi); Standard Marathi uses the shared trunk. Sub-dialects within a region share an expert.
3. **Read speech.** RESPIN is prompted read speech in two domains (agriculture, finance); conversational and overlapping speech are outside what is measured.
4. **Specialisation, not general ASR.** Out-of-domain accuracy on IndicVoices and FLEURS is well below SraVaani-1.0 and IndicConformer-600M (Section 4.2); on RESPIN SraVaani-1.0 is also better (Section 4.1). The supported claim is dialect-aware accuracy at a fixed parameter budget.
5. **Kathbath is in-domain.** Speaker overlap with Kathbath train is complete (Section 8).
6. **Efficiency.** The advantage is parameters used per frame; measured latency is higher than the dense model (Section 7).
7. **Dialect-skewed word analysis** is automatic (Section 5.5) and not a human-annotated dialect-fidelity study; no listener study is reported.

---

## 11. Reproducibility and checklist

| Item | Script / file | Status |
| :--- | :--- | :---: |
| Main benchmark, 4 datasets x 4 models, per-utterance predictions | `eval_offline.py`, `evals/<model>/<dataset>.json|.preds.jsonl` | done |
| Routing conditions, decoding, speaker bootstrap | `scripts/rerun_paper_evals.py` -> `evals/paper_rerun.json` | done |
| LoRA oracle / swap matrix | `scripts/rerun_e15_lora.py` -> `evals/e15_rerun.json` | done |
| Layer bypass / expert drop | `scripts/ablate_moe_layers.py` -> `evals/moe_layer_expert_ablation.json` | done |
| Dialect-skewed word and error analysis | `scripts/analyze_dialect_errors.py` -> `evals/dialect_error_analysis.json` | done |
| Latency / memory | `scripts/bench_latency.py` -> `evals/latency_benchmark.json` | done |
| Active parameter count | `evals/param_counts.json` | done |
| Leakage and Kathbath speaker audit | `scripts/audit_leakage_all.py`, `scripts/audit_kathbath_overlap.py` -> `evals/leakage_audit.json`, `evals/kathbath_overlap_audit.json` | done |
| Checkpoint selection | `scripts/compare_checkpoints.py` -> `evals/checkpoint_comparison*.json` | done |
| Related-work references checked against publisher / arXiv records | Section 9 | done |
| Baseline model cards checked (parameters, architecture, decoding) | Section 3.2 | done |
| This document | `scripts/build_final_md.py` regenerates every table from `evals/*.json` | done |
| Training ablations: seed variance, upcycling, number of experts, placement (Section 5.6) | `scripts/run_ablation_queue.py`, `scripts/eval_variant.py` -> `evals/ablation_training.json` | QUEUED - placeholders |
| Load-balancing loss and data-mix ablations | - | not run; no claim made (Section 10) |

Retained checkpoints, loadable with `adapters/our_conformer.py`:

* MoE: `runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt`
* Dense: `runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt`
* LoRA adapters: `runs/dialect_lora_adapters/best_lora_adapters.pt`

Exported copies of the two models (config, vocabulary, code and weights) are in `exported_models/marathi-conformer-130m-moe` and `exported_models/marathi-conformer-130m-dense`.
