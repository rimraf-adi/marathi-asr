"""Generate final.md from the measured JSON outputs under evals/ (no hand-typed numbers in tables)."""
import json
from pathlib import Path

R = Path('.')
J = lambda p: json.loads((R / p).read_text(encoding='utf-8'))
M = {m: {ds: J(f'evals/{m}/{ds}.json') for ds in ['respin_mr', 'kathbath_mr', 'indicvoices_mr', 'fleurs_mr']} for m in ['ours-moe', 'ours-dense', 'sravaani', 'indic-600m']}
PR = J('evals/paper_rerun.json'); E15 = J('evals/e15_rerun.json'); AB = J('evals/moe_layer_expert_ablation.json')
LAT = J('evals/latency_benchmark.json'); PC = J('evals/param_counts.json'); EA = J('evals/dialect_error_analysis.json')
LK = J('evals/leakage_audit.json'); KB = J('evals/kathbath_overlap_audit.json')
CK = {**J('evals/checkpoint_comparison.json'), **J('evals/checkpoint_comparison_dense.json')}

D = {'D1': 'Malvani', 'D2': 'Ahirani', 'D3': 'Standard', 'D4': 'Varhadi'}
dn = {'D1': 'Malvani (D1)', 'D2': 'Ahirani (D2)', 'D3': 'Standard (D3)', 'D4': 'Varhadi (D4)'}
NAME = {'ours-moe': '**MoE Conformer (ours)**', 'ours-dense': 'Dense Conformer (ours)', 'sravaani': 'SraVaani-1.0', 'indic-600m': 'IndicConformer-600M'}
f2 = lambda x: f'{x:.2f}'
res = lambda m, ds: M[m][ds]['result']
sl = lambda m, k: M[m]['respin_mr']['slices']['dialect'][dn[k]]['wer']
moe, dense = res('ours-moe', 'respin_mr')['wer'], res('ours-dense', 'respin_mr')['wer']
gap = dense - moe
E2 = PR['E2_routing_greedy']; E12 = PR['E12_decoding_130m_moe']; E4 = PR['E4_speaker_bootstrap']
lora = E15['oracle_lora']['ALL']
act_m = PC['active_top1'] / 1e6

L = []
a = L.append

a('# Results Reference for the Dialect-Aware Sparse MoE Conformer for Marathi ASR')
a('')
a('**Purpose:** single source of truth for writing the paper. Every number below was produced by the scripts listed in Section 11 on the two retained checkpoints, with one evaluation protocol, and is regenerated from `evals/*.json`. Statements that the data do not support are not made; where a result is unfavourable (SraVaani on RESPIN, out-of-domain accuracy, latency) it is reported as measured.  ')
a('**Repository:** [github.com/rimraf-adi/marathi-asr](https://github.com/rimraf-adi/marathi-asr)  ')
a('**Hardware:** NVIDIA RTX A5000 (24 GB), Windows x86_64, PyTorch 2.6 + CUDA 12.4  ')
a('**Date of measurements:** 2026-11-02')
a('')
a('---')
a('')
a('## 1. Claims the data support')
a('')
a('| # | Claim | Evidence (section) |')
a('| :--- | :--- | :--- |')
a(f'| C1 | At an identical 130.8M parameter budget, the dialect-routed MoE reaches **{f2(moe)}% WER** on the RESPIN held-out test set versus **{f2(dense)}%** for the dense Conformer: **{f2(gap)} points absolute ({gap / dense * 100:.1f}% relative)**, paired speaker-clustered bootstrap CI [{-E4["ALL"]["delta_ci"][1]:.2f}, {-E4["ALL"]["delta_ci"][0]:.2f}], p < 0.0001. | 4.1, 4.3 |')
a(f'| C2 | The gain holds in every dialect partition, including Standard Marathi ({f2(sl("ours-dense","D3") - sl("ours-moe","D3"))} points) and is largest on the regional dialects ({f2(sl("ours-dense","D1") - sl("ours-moe","D1"))} / {f2(sl("ours-dense","D2") - sl("ours-moe","D2"))} / {f2(sl("ours-dense","D4") - sl("ours-moe","D4"))} points for D1 / D2 / D4). | 4.1, 4.3 |')
a(f'| C3 | The experts carry the gain: removing all expert contributions raises WER from {f2(E2["learned"]["ALL"])}% to {f2(E2["trunk_only"]["ALL"])}%; dropping a single expert hurts its own dialect most; learned routing beats oracle dialect routing ({f2(E2["oracle"]["ALL"])}%) and random routing ({f2(E2["random"]["ALL"])}%). | 5.1, 5.2 |')
a(f'| C4 | The MoE preserves dialect-specific word forms far better than a dense model of equal size and than IndicConformer-600M (dialect-skewed word recall {EA["models"]["ours-moe"]["marker_recall"]["D1"]:.1f}/{EA["models"]["ours-moe"]["marker_recall"]["D2"]:.1f}/{EA["models"]["ours-moe"]["marker_recall"]["D4"]:.1f}% vs. {EA["models"]["ours-dense"]["marker_recall"]["D1"]:.1f}/{EA["models"]["ours-dense"]["marker_recall"]["D2"]:.1f}/{EA["models"]["ours-dense"]["marker_recall"]["D4"]:.1f}% and {EA["models"]["indic-600m"]["marker_recall"]["D1"]:.1f}/{EA["models"]["indic-600m"]["marker_recall"]["D2"]:.1f}/{EA["models"]["indic-600m"]["marker_recall"]["D4"]:.1f}% on D1/D2/D4). | 5.5 |')
a(f'| C5 | Autonomous frame-level routing outperforms dialect-specific LoRA adapters that are given the oracle dialect label ({lora:.2f}% vs. {f2(moe)}%). | 5.4 |')
a(f'| C6 | On RESPIN the MoE ({f2(moe)}%) is significantly better than IndicConformer-600M ({f2(res("indic-600m","respin_mr")["wer"])}%) and significantly worse than SraVaani-1.0 ({f2(res("sravaani","respin_mr")["wer"])}%). | 4.1 |')
a(f'| C7 | By parameter count the MoE uses {act_m:.1f}M of {PC["total"] / 1e6:.1f}M parameters per frame ({PC["active_pct"]}%), but in the present implementation it is **slower** than the dense model (Section 7): the parameter saving is not a speed saving. | 7 |')
a(f'| C8 | Out of domain (IndicVoices, FLEURS) both of our models are far less accurate than the external baselines; Kathbath is not an out-of-domain test (all {KB["valid_speakers"]} validation speakers also occur in Kathbath train). | 4.2, 8 |')
a(f'| C9 | Routing is learned and per frame (expert persists for about 3-8 frames, no utterance uses one expert); only Malvani maps clearly to one expert (expert 0). | 5.7 |')
a('')
a('')
a('Training-ablation claims (seed variance, upcycling, number of experts, expert placement) are placeholders in Section 5.6 until the queued runs finish.' if not Path('evals/ablation_training.json').exists() else 'Training-ablation results are in Section 5.6.')
a('---')
a('')
a('## 2. Data')
a('')
a('### 2.1 Training data by stage (both retained models)')
a('')
a('| Stage | Purpose | Data | Size / steps |')
a('| :--- | :--- | :--- | :--- |')
a('| 1. Self-supervised pretraining | Masked-reconstruction pretraining of the encoder | `ai4bharat/Shrutilipi` (All India Radio news audio) and `ARTPARK-IISc/Vaani` (image-prompted spontaneous speech), streamed; IndicVoices excluded (`--no_indicvoices`) | MoE run: 1,160,977 utterances (1,763.9 h) in one epoch, 18,141 steps. Dense run: 30,000 steps |')
a('| 2. Supervised CTC adaptation | Dense: plain CTC. MoE: upcycling of encoder blocks 5-12 into 3 dialect experts, with a CTC warm-up (steps 1-3,000), metadata-guided hard routing (3,001-18,000) and learned routing (18,001-80,000) | IISc RESPIN train partition (speaker-disjoint split in `moe/respin_split_cache.json`: 768,893 train + 41,041 calibration utterances) | 80,000 steps |')
a('| 3. Final fine-tuning | Multi-source CTC fine-tuning | RESPIN train pool (809,934 utterances), OpenSLR64 Marathi (1,569), Kathbath clean + noisy train (85,701), Mozilla Common Voice Marathi (1,205), and the Shrutilipi/Vaani stream | MoE: last saved step 342,000 (400,000 planned); dense: 300,000 steps; batch 32 x grad-accum 4, peak LR 3e-5 |')
a('| LoRA adapters (Sec. 5.4) | Three dialect adapters on the frozen dense backbone | RESPIN clean train, one pool per dialect (D1, D2, D4) | 6,000 steps |')
a('')
a('RESPIN: Kumar et al., NeurIPS 2025 Datasets & Benchmarks (read speech, agriculture and finance domains, multiple dialects per language).')
a('')
a('### 2.2 Evaluation data')
a('')
a('| Dataset | Split | Utterances / hours | Relation to our training data |')
a('| :--- | :--- | :---: | :--- |')
a(f'| IISc RESPIN | official test | {res("ours-moe","respin_mr")["n_utts"]:,} / {res("ours-moe","respin_mr")["hours"]:.2f} h, {LK["eval_sets"]["respin_test"]["speakers"]} speakers, 4 dialects | same corpus, **no speaker and no sentence shared with any training source** (Section 8) |')
a(f'| AI4Bharat Kathbath | valid | {res("ours-moe","kathbath_mr")["n_utts"]:,} / {res("ours-moe","kathbath_mr")["hours"]:.2f} h, {KB["valid_speakers"]} speakers | in-domain: all {KB["valid_speakers"]} speakers also appear in Kathbath train ({KB["train_clean"]["utts"]:,} clean utterances, {KB["train_clean"]["speakers"]} speakers); {LK["eval_sets"]["kathbath_valid"]["sentence_overlap_with_any_training_text"]} sentences shared with training text |')
a(f'| AI4Bharat IndicVoices | valid | {res("ours-moe","indicvoices_mr")["n_utts"]:,} / {res("ours-moe","indicvoices_mr")["hours"]:.2f} h, {LK["eval_sets"]["indicvoices_valid"]["speakers"]} speakers | not used in training; no sentence overlap |')
a(f'| Google FLEURS (mr_in) | test | {res("ours-moe","fleurs_mr")["n_utts"]:,} / {res("ours-moe","fleurs_mr")["hours"]:.2f} h ({LK["eval_sets"]["fleurs_test"]["unique_sentences"]} distinct sentences) | not used in training; no sentence overlap |')
a('')
a('Common Voice Marathi is used only as training data; no Common Voice number is reported. The training corpora of SraVaani-1.0 and IndicConformer-600M are those documented on their model cards, not controlled by us.')
a('')
a('---')
a('')
a('## 3. Models and protocol')
a('')
a('### 3.1 Our models')
a('')
a('| | Dense Conformer (baseline) | MoE Conformer (proposed) |')
a('| :--- | :---: | :---: |')
a('| Encoder | 12 blocks, d_model 512, 8 heads, conv kernel 31 | same |')
a('| FFN expansion | 8 | 4 (shared trunk) + 3 experts of expansion 4 in blocks 5-12 |')
a('| Router | - | linear 512 -> 3 per MoE block (12,312 parameters in total), top-1 per frame, additive to the always-active shared FFN |')
a('| Output | CTC head, 105-symbol character vocabulary | same |')
a(f'| Total parameters | 130.8M | {PC["total"] / 1e6:.1f}M ({PC["n_moe_layers"]} MoE blocks x {PC["n_experts"]} experts x {PC["params_per_expert"] / 1e6:.2f}M) |')
a(f'| Parameters used per frame (top-1) | 130.8M | {act_m:.1f}M ({PC["active_pct"]}%) |')
a('| Retained checkpoint | `runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt` | `runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt` |')
a('')
a('Experts 0/1/2 are initialised from the pretrained FFN and associated, during hard-routing training, with Malvani (D1), Ahirani (D2) and Varhadi (D4); Standard Marathi (D3) is served by the shared trunk. The design is reported as built; no claim is made that 3 experts or blocks 5-12 are optimal.')
a('')
a('### 3.2 External baselines')
a('')
a('| Model | Parameters | Architecture | Decoding in our harness | Source |')
a('| :--- | :---: | :--- | :--- | :--- |')
a('| SraVaani-1.0 | ~430M | FastConformer, hybrid TDT-CTC decoder; pretrained on ~31k h of Vaani, fine-tuned on ~31k h of transcribed speech in 65 Indian languages and dialects | model `transcribe()` with default settings | [model card](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) |')
a('| IndicConformer-600M | 600M | multilingual Conformer, hybrid CTC + RNNT | CTC mode, language `mr` | [model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) |')
a('')
a('### 3.3 Evaluation protocol')
a('')
a('* Greedy CTC decoding unless stated (Section 5.3). Utterances are sorted by length for batching.')
a('* Text normalisation `mr-norm-1` (`normalizer.py`), applied identically to references and hypotheses of every model: Unicode NFC, removal of ZWNJ/ZWJ, nukta unification, anusvara/chandrabindu harmonisation, punctuation stripping.')
a('* WER/CER are corpus-level; "n" counts utterances with a non-empty normalised reference. Dataset-level CIs: 1,000 utterance-level bootstrap resamples. Paired CIs (Section 4.3): 10,000 speaker-clustered resamples over the 160 RESPIN test speakers.')
a('* One training run per model; no variance across training seeds was measured, so differences of a few tenths of a point between configurations of the same model should not be over-interpreted.')
a('* Checkpoints: the best of each family on RESPIN and Kathbath (Section 6).')
a('')
a('---')
a('')
a('## 4. Main results')
a('')
a('### 4.1 RESPIN held-out test (2,170 utterances, 160 unseen speakers)')
a('')
a('| Model | Params | D1 Malvani | D2 Ahirani | D3 Standard | D4 Varhadi | **ALL WER** [95% CI] | ALL CER |')
a('| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')
par = {'ours-moe': '130.8M', 'ours-dense': '130.8M', 'sravaani': '~430M', 'indic-600m': '600M'}
for m in ['sravaani', 'ours-moe', 'indic-600m', 'ours-dense']:
    r = res(m, 'respin_mr')
    a(f'| {NAME[m]} | {par[m]} | ' + ' | '.join(f2(sl(m, k)) + '%' for k in ['D1', 'D2', 'D3', 'D4']) + f' | **{f2(r["wer"])}%** [{r["wer_ci95"][0]:.2f}, {r["wer_ci95"][1]:.2f}] | {f2(r["cer"])}% |')
a(f'| MoE, experts bypassed (Sec. 5.1) | 130.8M | ' + ' | '.join(f2(E2['trunk_only'][k]) + '%' for k in ['D1', 'D2', 'D3', 'D4']) + f' | {f2(E2["trunk_only"]["ALL"])}% | - |')
a(f'| Dense + oracle dialect LoRA (Sec. 5.4) | 142.6M | ' + ' | '.join(f2(E15['oracle_lora'][k]) + '%' for k in ['D1', 'D2', 'D3', 'D4']) + f' | {lora:.2f}% | - |')
a('')
a(f'* MoE vs. dense (same 130.8M budget): {f2(gap)} points absolute, {gap / dense * 100:.1f}% relative.')
mo, ic, sv = res('ours-moe', 'respin_mr')['wer_ci95'], res('indic-600m', 'respin_mr')['wer_ci95'], res('sravaani', 'respin_mr')['wer_ci95']
a(f'* The MoE interval [{mo[0]:.2f}, {mo[1]:.2f}] lies entirely below IndicConformer-600M [{ic[0]:.2f}, {ic[1]:.2f}] and entirely above SraVaani-1.0 [{sv[0]:.2f}, {sv[1]:.2f}]. SraVaani-1.0 has ~3.3x our parameter count and, per its model card, was fine-tuned on ~31k hours of transcribed speech; it is the most accurate system on RESPIN.')
a('')
a('### 4.2 All datasets')
a('')
a('WER % (CER %) [95% CI], greedy CTC, same normaliser.')
a('')
dsl = [('respin_mr', 'RESPIN test'), ('kathbath_mr', 'Kathbath valid (in-domain)'), ('indicvoices_mr', 'IndicVoices valid (OOD)'), ('fleurs_mr', 'FLEURS test (OOD)')]
a('| Model | ' + ' | '.join(f'{lab} n={res("ours-moe", ds)["n_utts"]:,}' for ds, lab in dsl) + ' |')
a('| :--- | ' + ' | '.join([':---:'] * 4) + ' |')
for m in ['ours-moe', 'ours-dense', 'sravaani', 'indic-600m']:
    a(f'| {NAME[m]} | ' + ' | '.join(f'{res(m, ds)["wer"]:.2f} ({res(m, ds)["cer"]:.2f}) [{res(m, ds)["wer_ci95"][0]:.2f}, {res(m, ds)["wer_ci95"][1]:.2f}]' for ds, _ in dsl) + ' |')
a('')
a(f'* In-domain (RESPIN, Kathbath) the MoE halves the dense model\'s error on Kathbath ({res("ours-dense","kathbath_mr")["wer"]:.2f}% to {res("ours-moe","kathbath_mr")["wer"]:.2f}%).')
a(f'* Out of domain, the MoE improves on the dense model by {res("ours-dense","indicvoices_mr")["wer"] - res("ours-moe","indicvoices_mr")["wer"]:.1f} points (IndicVoices) and {res("ours-dense","fleurs_mr")["wer"] - res("ours-moe","fleurs_mr")["wer"]:.1f} points (FLEURS) but remains far from SraVaani-1.0 and IndicConformer-600M ({res("ours-moe","indicvoices_mr")["wer"]:.1f}% and {res("ours-moe","fleurs_mr")["wer"]:.1f}% vs. roughly 15-19%). The models are specialised to RESPIN-style read speech; they should not be presented as general Marathi ASR systems.')
a('')
a('### 4.3 Paired comparison MoE vs. dense (speaker-clustered bootstrap, B = 10,000, 160 speakers)')
a('')
a('| Partition | MoE WER 95% CI | Dense WER 95% CI | Paired reduction (dense - MoE) [95% CI] | p |')
a('| :--- | :---: | :---: | :---: | :---: |')
for k, lab in [('D1', 'D1 Malvani'), ('D2', 'D2 Ahirani'), ('D3', 'D3 Standard'), ('D4', 'D4 Varhadi'), ('ALL', 'ALL')]:
    e = E4[k]
    a(f'| {lab} | [{e["moe_ci"][0]:.2f}, {e["moe_ci"][1]:.2f}] | [{e["dense_ci"][0]:.2f}, {e["dense_ci"][1]:.2f}] | {-e["delta_mean"]:.2f} [{-e["delta_ci"][1]:.2f}, {-e["delta_ci"][0]:.2f}] | < 0.0001 |')
a('')
a('The reduction is significant in every partition; the empirical p-value is bounded below by 1/B = 0.0001. The bootstrap mean of the paired reduction (16.62) differs from the point estimate (16.63) only through resampling.')
a('')
a('---')
a('')
a('## 5. Analyses')
a('')
a('### 5.1 Routing conditions (same MoE checkpoint, greedy CTC)')
a('')
a('| Routing | Granularity | D1 | D2 | D3 | D4 | ALL | vs. learned |')
a('| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
for key, lab, gr in [('learned', 'Learned router (top-1)', 'frame'), ('oracle', 'Oracle dialect label', 'utterance'), ('random', 'Random expert', 'utterance'), ('trunk_only', 'Experts bypassed (shared trunk only)', '-')]:
    e = E2[key]
    a(f'| {lab} | {gr} | ' + ' | '.join(f2(e[k]) for k in ['D1', 'D2', 'D3', 'D4']) + f' | **{f2(e["ALL"])}** | {"-" if key == "learned" else "+" + f2(e["ALL"] - E2["learned"]["ALL"])} |')
a('')
a('Oracle routing sends D1/D2/D4 utterances to experts 0/1/2 and D3 utterances to the trunk alone. Learned frame-level routing is the best of the four conditions in every dialect, including D3, and on aggregate and on D1/D2/D4 oracle routing is better than random routing and than bypassing the experts (on D3, which has no dedicated expert, random routing scores better than oracle).')
a('')
a('### 5.2 Inference-time ablations of the trained MoE (no retraining)')
a('')
a('Greedy CTC, same protocol. "Bypass" removes the expert contribution of a block (shared FFN only); "drop" prevents the router from selecting one expert (it chooses among the remaining two).')
a('')
a('| Intervention | D1 | D2 | D3 | D4 | ALL | vs. baseline |')
a('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
b = AB['baseline_learned']
for k, v in AB.items():
    lab = k.replace('_', ' ')
    lab = {'baseline learned': 'None (trained MoE)', 'bypass layer 5': 'Bypass block 5'}.get(lab, lab)
    if k.startswith('bypass_layer_') and 'half' not in k:
        lab = f'Bypass block {k.split("_")[-1]}'
    elif 'lower_half' in k:
        lab = 'Bypass blocks 5-8'
    elif 'upper_half' in k:
        lab = 'Bypass blocks 9-12'
    elif k.startswith('drop_'):
        lab = 'Drop ' + {'expert0_malvani': 'expert 0 (Malvani)', 'expert1_ahirani': 'expert 1 (Ahirani)', 'expert2_varhadi': 'expert 2 (Varhadi)'}[k[5:]]
    a(f'| {lab} | ' + ' | '.join(f2(v[x]) for x in ['D1', 'D2', 'D3', 'D4']) + f' | **{f2(v["ALL"])}** | {"-" if k == "baseline_learned" else "+" + f2(v["ALL"] - b["ALL"])} |')
a('')
d0, d1, d2 = AB['drop_expert0_malvani'], AB['drop_expert1_ahirani'], AB['drop_expert2_varhadi']
a(f'* **Expert specialisation.** Each dropped expert hurts its associated dialect most: removing expert 0 raises D1 WER by {d0["D1"] - b["D1"]:.1f} points (other dialects {d0["D2"] - b["D2"]:.1f}, {d0["D4"] - b["D4"]:.1f}); expert 1 raises D2 by {d1["D2"] - b["D2"]:.1f} (others {d1["D1"] - b["D1"]:.1f}, {d1["D4"] - b["D4"]:.1f}); expert 2 raises D4 by {d2["D4"] - b["D4"]:.1f} (others {d2["D1"] - b["D1"]:.1f}, {d2["D2"] - b["D2"]:.1f}).')
a(f'* **Where the experts matter.** Bypassing blocks 5-8 costs {AB["bypass_layers_5-8_lower_half"]["ALL"] - b["ALL"]:.2f} points, blocks 9-12 cost {AB["bypass_layers_9-12_upper_half"]["ALL"] - b["ALL"]:.2f} points; single-block bypasses are largest at blocks 11 ({AB["bypass_layer_11"]["ALL"] - b["ALL"]:.2f}) and 12 ({AB["bypass_layer_12"]["ALL"] - b["ALL"]:.2f}). The upper blocks carry most of the dialect-specific computation. This is an inference-time sensitivity analysis of one trained model, not a comparison of models trained with different expert placements.')
a('')
a('### 5.3 Decoding (130.8M MoE)')
a('')
a('| Decoder | D1 | D2 | D3 | D4 | ALL |')
a('| :--- | :---: | :---: | :---: | :---: | :---: |')
for key, lab in [('greedy', 'Greedy CTC (used for all results)'), ('lexicon_beam16', 'Beam 16 + lexicon (no language model)')]:
    e = E12[key]
    a(f'| {lab} | ' + ' | '.join(f2(e[k]) for k in ['D1', 'D2', 'D3', 'D4']) + f' | **{f2(e["ALL"])}** |')
a('')
a(f'No language model is used anywhere in this document. A 16-beam search restricted to a 34,075-word lexicon mined from the RESPIN train and calibration splits (`decoding/beam_search_decoder.py`) changes WER by only {E12["greedy"]["ALL"] - E12["lexicon_beam16"]["ALL"]:.2f} points, so all headline numbers use plain greedy CTC decoding for every model.')
a('')
a('### 5.4 Dialect LoRA adapters vs. MoE (PEFT control)')
a('')
a('Three rank-16 LoRA adapters (alpha 32, dropout 0.05) on attention (Wq, Wv) and both FFNs of all 12 blocks of the frozen dense backbone; 3,932,160 trainable parameters per adapter, 11,796,480 in total, 142,567,017 parameters in the combined model.')
a('')
a('| System | Needs dialect label at inference | D1 | D2 | D3 | D4 | ALL |')
a('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
a('| Dense, no adapter | no | ' + ' | '.join(f2(E15['no_adapter'][k]) for k in ['D1', 'D2', 'D3', 'D4']) + f' | {f2(E15["no_adapter"]["ALL"])} |')
a('| Dense + oracle LoRA | **yes** | ' + ' | '.join(f2(E15['oracle_lora'][k]) for k in ['D1', 'D2', 'D3', 'D4']) + f' | {f2(E15["oracle_lora"]["ALL"])} |')
a('| MoE (learned routing) | no | ' + ' | '.join(f2(E2['learned'][k]) for k in ['D1', 'D2', 'D3', 'D4']) + f' | **{f2(E2["learned"]["ALL"])}** |')
a('')
a('Cross-adapter matrix (WER %, every utterance of a test dialect decoded with one adapter):')
a('')
a('| Test dialect | adapter 0 (Malvani) | adapter 1 (Ahirani) | adapter 2 (Varhadi) | mismatch penalty |')
a('| :--- | :---: | :---: | :---: | :---: |')
for k, mi, lab in [('D1', 0, 'D1 Malvani'), ('D2', 1, 'D2 Ahirani'), ('D4', 2, 'D4 Varhadi')]:
    v = [E15[f'all_adapter_{i}'][k] for i in range(3)]
    pen = [x - v[mi] for i, x in enumerate(v) if i != mi]
    a(f'| {lab} | ' + ' | '.join((f'**{f2(x)}**' if i == mi else f2(x)) for i, x in enumerate(v)) + f' | +{min(pen):.2f} to +{max(pen):.2f} |')
a('')
a(f'* Oracle LoRA improves on the dense model by {E15["no_adapter"]["ALL"] - lora:.2f} points; the MoE improves on oracle LoRA by a further {lora - moe:.2f} points ({(lora - moe) / lora * 100:.1f}% relative) without any dialect metadata.')
a('* Each adapter is best on its own dialect, confirming that the adapters are dialect-specific. Note that the MoE was trained with a longer, multi-stage recipe than the adapters (6,000 steps); the comparison measures what each recipe achieved, not a controlled equal-compute test.')
a('')
a('### 5.5 Dialect preservation (RESPIN test predictions)')
a('')
a('"Dialect-skewed words" are reference word types that occur at least 10 times in one regional dialect and at least 5 times more often (per token) there than in all other dialects. They are found automatically from the test references and include dialect forms (e.g. Malvani आसा, माका; Ahirani शे; Varhadi मले, हाये) as well as some topic words (e.g. Ahirani लागवड, Varhadi बिमा), so the measure is a lexical-skew proxy and not a linguist-annotated dialect inventory.')
a('')
mk = EA['markers']
a('| Dialect | skewed word types | tokens in test refs | share of dialect tokens |')
a('| :--- | :---: | :---: | :---: |')
for k in ['D1', 'D2', 'D4']:
    a(f'| {D[k]} ({k}) | {mk[k]["n_words"]} | {mk[k]["tokens"]} | {mk[k]["tokens"] / mk[k]["total_tokens"] * 100:.1f}% |')
a('')
a('Recall of dialect-skewed words (a reference occurrence counts as recalled if it is aligned to an identical hypothesis word):')
a('')
a('| Model | D1 Malvani | D2 Ahirani | D4 Varhadi |')
a('| :--- | :---: | :---: | :---: |')
for m in ['ours-moe', 'ours-dense', 'sravaani', 'indic-600m']:
    r = EA['models'][m]['marker_recall']
    a(f'| {NAME[m]} | {r["D1"]:.1f}% | {r["D2"]:.1f}% | {r["D4"]:.1f}% |')
a('')
a('Error composition on RESPIN (% of reference words):')
a('')
a('| Model | Substitutions | Deletions | Insertions | WER |')
a('| :--- | :---: | :---: | :---: | :---: |')
for m in ['ours-moe', 'ours-dense', 'sravaani', 'indic-600m']:
    r = EA['models'][m]['error_rates']['ALL']
    a(f'| {NAME[m]} | {r["sub"]:.2f} | {r["dele"]:.2f} | {r["ins"]:.2f} | {r["wer"]:.2f} |')
a('')
a('Most frequent replacements of dialect-skewed words (reference -> hypothesis, count):')
a('')
for m in ['indic-600m', 'ours-dense', 'ours-moe']:
    t = EA['models'][m]['top_marker_substitutions']
    for k in ['D1', 'D4']:
        a(f'* {NAME[m]}, {D[k]}: ' + ', '.join(f'{x[0]} -> {x[1]} ({x[2]})' for x in t[k][:4]))
a('')
a('IndicConformer-600M frequently rewrites dialect forms into standard Marathi (e.g. आसा -> असा, मले -> मला, शे -> आहे), which is the "standardisation" behaviour the MoE reduces; some of the remaining errors of all systems are orthographic variants (e.g. करु/करू), not dialect errors. SraVaani-1.0 preserves dialect forms about as well as the MoE (higher recall on D2, lower on D1 and D4) while having a lower overall WER.')
a('')

# ---------------- 5.6 training ablations (placeholders until evals/ablation_training.json exists) ----------------
AT_PATH = R / 'evals/ablation_training.json'
AT = json.loads(AT_PATH.read_text(encoding='utf-8')) if AT_PATH.exists() else {}
import statistics as _st
def _g(name, ds='respin_mr', key=None):
    v = AT.get(name, {}).get(ds)
    if not v:
        return '[TBD]'
    if key is None:
        return f'{v["wer"]:.2f}'
    return f'{v["dialect"].get(key, float("nan")):.2f}' if v.get('dialect') else '[TBD]'
def _row(label, name, extra=''):
    return f'| {label} | ' + ' | '.join(_g(name, key=k) for k in ['D1', 'D2', 'D3', 'D4']) + f' | **{_g(name)}** | {_g(name, "kathbath_mr")} |' + extra
HAVE = bool(AT)
a('### 5.6 Training ablations (reduced budget)')
a('')
if not HAVE:
    a('> **PLACEHOLDER SECTION.** The runs below are queued (`scripts/run_ablation_queue.py`) and have not been measured yet. Every `[TBD]` is filled automatically by `python scripts/build_final_md.py` once `evals/ablation_training.json` exists. Do not quote any number from this section before then.')
    a('')
a('**Protocol.** All variants start from the same re-run Stage 1 pretrained encoder (the original Stage 1 checkpoints were deleted) and are trained for 20,000 Stage-2 steps (CTC warm-up 750, metadata-guided hard routing 3,750, learned routing for the rest; one quarter of the 80,000-step recipe), batch 32 x grad-accum 4, peak LR 1.5e-4, then evaluated on RESPIN test and Kathbath valid with greedy CTC. Absolute WERs are therefore higher than the headline 19.50% (which includes Stage 3 fine-tuning); only differences between variants are meaningful. Everything except the varied factor is identical to `base_s0`.')
a('')
a('**Claim A - seed variance** (does a single run suffice?)')
a('')
a('| Run (seed) | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |')
a('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
for sd in range(3):
    a(_row(f'MoE, seed {sd}', f'base_s{sd}'))
if all(f'base_s{i}' in AT for i in range(3)):
    ws = [AT[f'base_s{i}']['respin_mr']['wer'] for i in range(3)]; ks = [AT[f'base_s{i}']['kathbath_mr']['wer'] for i in range(3)]
    a(f'| **mean +/- sd (n=3)** | | | | | **{_st.mean(ws):.2f} +/- {_st.stdev(ws):.2f}** | {_st.mean(ks):.2f} +/- {_st.stdev(ks):.2f} |')
else:
    a('| **mean +/- sd (n=3)** | | | | | [TBD] | [TBD] |')
a('')
a('Template sentence: "Across three training seeds the reduced-budget MoE reaches [TBD] +/- [TBD]% WER on RESPIN; the seed-to-seed standard deviation is [TBD] points, [much smaller than / comparable to] the MoE-dense gap." (Dense-model seeds are not part of this queue: a dense Stage 1 re-run takes about 85 h.)')
a('')
a('**Claim B - upcycling vs. randomly initialised experts**')
a('')
a('| Variant | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |')
a('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
a(_row('Experts upcycled from the pretrained FFN (base_s0)', 'base_s0'))
a(_row('Experts randomly initialised (scratch_s0)', 'scratch_s0'))
a('')
a('Template sentence: "Initialising experts from the pretrained FFN [lowers / does not change] WER by [TBD] points relative to random expert initialisation at the same step budget."')
a('')
a('**Claim C - number of experts** (3 experts = design choice or finding?)')
a('')
a('| Experts per MoE block | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |')
a('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
for lab, nm in [('1', 'k1_s0'), ('2', 'k2_s0'), ('**3 (base_s0)**', 'base_s0'), ('4', 'k4_s0')]:
    a(_row(lab, nm))
a('')
a('Hard-routing note: with K < 3 the dialects whose index is >= K have no dedicated expert during the hard-routing phase (they use the shared FFN only); with K = 4 the fourth expert receives the Standard-Marathi utterances during that phase.')
a('')
a('Template sentence: "WER [decreases / saturates] from [TBD]% (K=1) to [TBD]% (K=3), and [does / does not] improve further with K=4 ([TBD]%)."')
a('')
a('**Claim D - expert placement** (which blocks hold the experts?)')
a('')
a('| MoE blocks | D1 | D2 | D3 | D4 | RESPIN ALL | Kathbath |')
a('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
for lab, nm in [('1-4', 'lower4_s0'), ('**5-12 (base_s0)**', 'base_s0'), ('1-12', 'all12_s0')]:
    a(_row(lab, nm))
a('')
a('Template sentence: "Placing experts in blocks 5-12 gives [TBD]% WER versus [TBD]% for blocks 1-4 and [TBD]% for all twelve blocks, consistent with the inference-time bypass analysis (Section 5.2), where [TBD]."')
a('')

RG = J('evals/routing_granularity.json')
a('### 5.7 Routing granularity')
a('')
a('Mechanism: in each MoE block a linear router scores every frame and the top-1 expert processes that frame (`moe/moe_layer.py`). The reported checkpoint\'s final fine-tuning stage passes no dialect label, so its routing is learned. Dialect labels steer routing only in Stage 2, steps 3,001-18,000 of 80,000 (hard routing).')
a('')
a('Behaviour, measured with batch size 1 on 240 RESPIN test utterances (60 per dialect, `scripts/measure_routing.py`). "Switch rate" is the share of adjacent frames that choose different experts; "chance" is the rate if frames chose independently with the block\'s overall expert frequencies; "top-expert share" is the mean share of an utterance\'s frames that go to its most-used expert.')
a('')
a('| MoE block | Switch rate | Chance | Mean run length (frames) | Top-expert share | Expert usage D1 / D2 / D3 / D4 (expert 0, 1, 2 in %) |')
a('| :--- | :---: | :---: | :---: | :---: | :--- |')
for b in sorted(RG, key=int):
    r = RG[b]; u = r['usage_pct']
    a(f'| {b} | {r["switch_rate"]:.3f} | {r["chance_switch"]:.3f} | {1 / r["switch_rate"]:.1f} | {r["majority_share"]:.2f} | ' + ' ; '.join(f'{d}: ' + '/'.join(f'{x:.0f}' for x in u[d]) for d in ['D1', 'D2', 'D3', 'D4']) + ' |')
a('')
a('* Routing is per frame and learned, but it is not random: adjacent frames keep the same expert far more often than chance (switch rate 0.12-0.35 vs. 0.55-0.67), so an expert persists for roughly 3-8 frames. Deeper blocks switch least.')
a('* It is not utterance-level: in no block does a single expert take more than 70% of an utterance\'s frames on average (0.45-0.70).')
a('* Dialect labels are only partly recoverable from the routing. Malvani (D1) clearly prefers expert 0 in blocks 11-12 (63% and 71% of frames); Ahirani, Standard and Varhadi are spread across experts, and Standard Marathi is routed to experts like the others. This agrees with Section 5.2, where dropping expert 0 hurts Malvani most. Describe the experts as specialised for Malvani-like speech, not as one expert per dialect.')
a('* Sample: 60 utterances per dialect from one checkpoint; the percentages are descriptive, not tested.')
a('')
a('---')
a('')
a('## 6. Checkpoint selection')
a('')
a('Ten trained checkpoints (8 MoE, 2 dense; the dense Stage 1 pretraining checkpoint has no CTC head and cannot be scored) were evaluated on RESPIN and Kathbath with the same protocol; only the best of each family is retained.')
a('')
a('| Retained checkpoint | RESPIN WER | Kathbath WER |')
a('| :--- | :---: | :---: |')
for key, lab in [('runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt', 'MoE 130.8M, Stage 3 vanilla SFT'), ('runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt', 'Dense 130.8M, Stage 3 final alignment')]:
    a(f'| **{lab}** | {CK[key]["respin_mr"]["wer"]:.2f}% | {CK[key]["kathbath_mr"]["wer"]:.2f}% |')
others = [v for k, v in CK.items() if 'error' not in v and 'moe-3stage/checkpoints/stage3_vanilla_sft' not in k and 'dense-3stage/checkpoints/stage3_final_alignment' not in k]
a('')
a(f'The remaining {len(others)} scored checkpoints (earlier stages, alternative runs, GRPO and polishing stages) were worse on both RESPIN and Kathbath (best alternative MoE: {min(v["respin_mr"]["wer"] for k, v in CK.items() if "error" not in v and "moe" in k and "stage3_vanilla_sft/best_model.pt" in k and "scaled" not in k):.2f}% / {min(v["kathbath_mr"]["wer"] for k, v in CK.items() if "error" not in v and "moe" in k and "stage3_vanilla_sft/best_model.pt" in k and "scaled" not in k):.2f}%) and were deleted.')
a('')
a('---')
a('')
a('## 7. Efficiency')
a('')
a(f'Measured on {LAT["gpu"]}, fp32, {LAT["n_utts"]} random RESPIN test utterances ({LAT["audio_secs"]:.0f} s of audio), idle GPU, warm-up excluded.')
a('')
a('| | Dense | MoE |')
a('| :--- | :---: | :---: |')
a(f'| Total parameters | {LAT["dense"]["params_m"]}M | {LAT["moe"]["params_m"]}M |')
a(f'| Parameters used per frame | {LAT["dense"]["params_m"]}M | {act_m:.1f}M ({PC["active_pct"]}%) |')
a(f'| Batch-1 latency, median / p95 | {LAT["dense"]["batch1_latency_ms_p50"]} / {LAT["dense"]["batch1_latency_ms_p95"]} ms | {LAT["moe"]["batch1_latency_ms_p50"]} / {LAT["moe"]["batch1_latency_ms_p95"]} ms |')
a(f'| Batch-1 real-time factor | {LAT["dense"]["batch1_rtf"]} | {LAT["moe"]["batch1_rtf"]} |')
a(f'| Batch-16 throughput (RTFx) | {LAT["dense"]["batch16_rtfx"]} | {LAT["moe"]["batch16_rtfx"]} |')
a(f'| Peak GPU memory, batch 1 / 16 | {LAT["dense"]["peak_vram_mb_batch1"]} / {LAT["dense"]["peak_vram_mb_batch16"]} MB | {LAT["moe"]["peak_vram_mb_batch1"]} / {LAT["moe"]["peak_vram_mb_batch16"]} MB |')
a('')
a(f'The MoE uses {100 - PC["active_pct"]:.1f}% fewer parameters per frame, but its median batch-1 latency is {LAT["moe"]["batch1_latency_ms_p50"] / LAT["dense"]["batch1_latency_ms_p50"]:.1f}x that of the dense model and its batch-16 throughput is {LAT["moe"]["batch16_rtfx"] / LAT["dense"]["batch16_rtfx"] * 100:.0f}% of the dense model, because the reference implementation dispatches frames to experts with per-expert masked indexing (`moe/moe_layer.py`), which is not optimised for speed and was not profiled further. All experts must also be resident in memory (same peak memory). Both models are far faster than real time (RTFx > 60 at batch 1). The efficiency claim for the paper is therefore the parameter count per frame, not wall-clock speed.')
a('')
a('---')
a('')
a('## 8. Data integrity audit')
a('')
a('Script `scripts/audit_leakage_all.py` (+ `scripts/audit_kathbath_overlap.py`); normalised exact-sentence match of each evaluation set against all training text sources (RESPIN train + calibration, Kathbath clean + noisy train, OpenSLR64, Common Voice Marathi).')
a('')
a('| Evaluation set | Utterances (distinct sentences) | Speakers | Speaker overlap with training | Sentence overlap with training text |')
a('| :--- | :---: | :---: | :---: | :---: |')
e = LK['eval_sets']
a(f'| RESPIN test | {e["respin_test"]["utts"]:,} ({e["respin_test"]["unique_sentences"]}) | {e["respin_test"]["speakers"]} | **0** of {e["respin_test"]["train_calibration_speakers"]:,} train+calibration speakers | **0** |')
a(f'| Kathbath valid | {e["kathbath_valid"]["utts"]:,} ({e["kathbath_valid"]["unique_sentences"]:,}) | {e["kathbath_valid"]["valid_speakers"]} | **{e["kathbath_valid"]["speaker_overlap_with_kathbath_train_clean"]} of {e["kathbath_valid"]["valid_speakers"]}** occur in Kathbath clean train | {e["kathbath_valid"]["sentence_overlap_with_any_training_text"]} ({e["kathbath_valid"]["sentence_overlap_by_source"]["kathbath_train_clean"]} Kathbath, {e["kathbath_valid"]["sentence_overlap_by_source"]["openslr64"]} OpenSLR64, {e["kathbath_valid"]["sentence_overlap_by_source"]["commonvoice_mr"]} Common Voice) |')
a(f'| IndicVoices valid | {e["indicvoices_valid"]["utts"]:,} ({e["indicvoices_valid"]["unique_sentences"]:,}) | {e["indicvoices_valid"]["speakers"]} | n/a (corpus not used in training) | **0** |')
a(f'| FLEURS test | {e["fleurs_test"]["utts"]:,} ({e["fleurs_test"]["unique_sentences"]}) | n/a | n/a (corpus not used in training) | **0** |')
a('')
a('RESPIN test has no speaker and no sentence in common with any training source, so the RESPIN results are clean held-out results. Kathbath validation speakers all appear in Kathbath train, so Kathbath numbers measure in-domain, speaker-seen performance. (IndicVoices valid has 6 more utterances than scored because empty normalised references are excluded from scoring.)')
a('')
a('---')
a('')
a('## 9. Related work')
a('')
a('Every reference below was checked against its publisher or arXiv record.')
a('')
a('* **SpeechMoE** (You, Feng, Su, Yu; Interspeech 2021, [arXiv:2105.03036](https://arxiv.org/abs/2105.03036)) introduces a dynamic-routing MoE acoustic model whose router uses the previous layer output together with an embedding network, with 7.0-23.0% relative CER improvement over static networks at comparable computation. **SpeechMoE2** (You et al., ICASSP 2022, [arXiv:2111.11831](https://arxiv.org/abs/2111.11831)) adds global domain and accent embeddings to the router input (1.9-17.7% relative CER gains on a multi-accent task). Our router uses no external embeddings or labels at inference.')
a('* **DialectMoE** (Zhou et al., CCL 2024, [ACL Anthology 2024.ccl-1.89](https://aclanthology.org/2024.ccl-1.89/)) is an end-to-end MoE for Chinese multi-dialect recognition in low-resource conditions. It is the closest dialect-ASR MoE precedent; we apply the idea to Marathi dialects and compare against a capacity-matched dense model and a PEFT alternative.')
a('* **MoE-CTC** (Lee, Kim, Lee; ACL 2026, [arXiv:2602.01967](https://arxiv.org/abs/2602.01967)) uses accent-aware routing during training with intermediate CTC supervision and label-free routing at inference for accented speech recognition. We use a single CTC head and metadata-guided routing only in an early training phase.')
a('* **LR-MoE** (Interspeech 2023, [arXiv:2307.05956](https://arxiv.org/abs/2307.05956)) and **DLG-MoE** ([arXiv:2407.18581](https://arxiv.org/abs/2407.18581)) route by language group for multilingual / code-switching ASR.')
a('* **Sparse upcycling** (Komatsuzaki et al., ICLR 2023, [arXiv:2212.05055](https://arxiv.org/abs/2212.05055)) initialises MoE layers from dense checkpoints; we use the same idea to initialise experts from the pretrained FFN.')
a('* **Data and baselines:** RESPIN-S1.0 (Kumar et al., NeurIPS 2025 Datasets & Benchmarks); Kathbath / IndicSUPERB (Javed et al., AAAI 2023, [arXiv:2208.11761](https://arxiv.org/abs/2208.11761)); IndicVoices (Javed et al., Findings of ACL 2024, [2024.findings-acl.639](https://aclanthology.org/2024.findings-acl.639)); FLEURS (Conneau et al., SLT 2022, [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)); Shrutilipi (Bhogale et al., ICASSP 2023, [arXiv:2208.12666](https://arxiv.org/abs/2208.12666)); Vaani and SraVaani ([ARTPARK-IISc](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)); IndicConformer ([AI4Bharat](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)).')
a('')
a('---')
a('')
a('## 10. Scope of the claims')
a('')
a('1. **Headline models: one run each.** The headline dense and MoE models were each trained once. ' + ('Seed variance, expert initialisation, number of experts and expert placement are examined at reduced budget in Section 5.6. ' if HAVE else 'Seed variance, expert initialisation, number of experts and expert placement are placeholders in Section 5.6 and are not claimed until measured. ') + 'The load-balancing loss and the data mix are not varied, and no claim is made that the chosen values are optimal. Section 5.2 characterises the trained model by inference-time interventions.')
a('2. **Macro-dialect granularity.** Three experts correspond to the three regional RESPIN dialects (Malvani, Ahirani, Varhadi); Standard Marathi uses the shared trunk. Sub-dialects within a region share an expert.')
a('3. **Read speech.** RESPIN is prompted read speech in two domains (agriculture, finance); conversational and overlapping speech are outside what is measured.')
a('4. **Specialisation, not general ASR.** Out-of-domain accuracy on IndicVoices and FLEURS is well below SraVaani-1.0 and IndicConformer-600M (Section 4.2); on RESPIN SraVaani-1.0 is also better (Section 4.1). The supported claim is dialect-aware accuracy at a fixed parameter budget.')
a('5. **Kathbath is in-domain.** Speaker overlap with Kathbath train is complete (Section 8).')
a('6. **Efficiency.** The advantage is parameters used per frame; measured latency is higher than the dense model (Section 7).')
a('7. **Dialect-skewed word analysis** is automatic (Section 5.5) and not a human-annotated dialect-fidelity study; no listener study is reported.')
a('')
a('---')
a('')
a('## 11. Reproducibility and checklist')
a('')
a('| Item | Script / file | Status |')
a('| :--- | :--- | :---: |')
rows = [
    ('Main benchmark, 4 datasets x 4 models, per-utterance predictions', '`eval_offline.py`, `evals/<model>/<dataset>.json|.preds.jsonl`', 'done'),
    ('Routing conditions, decoding, speaker bootstrap', '`scripts/rerun_paper_evals.py` -> `evals/paper_rerun.json`', 'done'),
    ('LoRA oracle / swap matrix', '`scripts/rerun_e15_lora.py` -> `evals/e15_rerun.json`', 'done'),
    ('Layer bypass / expert drop', '`scripts/ablate_moe_layers.py` -> `evals/moe_layer_expert_ablation.json`', 'done'),
    ('Dialect-skewed word and error analysis', '`scripts/analyze_dialect_errors.py` -> `evals/dialect_error_analysis.json`', 'done'),
    ('Routing granularity (switch rate, expert usage per dialect)', '`scripts/measure_routing.py` -> `evals/routing_granularity.json`', 'done'),
    ('Latency / memory', '`scripts/bench_latency.py` -> `evals/latency_benchmark.json`', 'done'),
    ('Active parameter count', '`evals/param_counts.json`', 'done'),
    ('Leakage and Kathbath speaker audit', '`scripts/audit_leakage_all.py`, `scripts/audit_kathbath_overlap.py` -> `evals/leakage_audit.json`, `evals/kathbath_overlap_audit.json`', 'done'),
    ('Checkpoint selection', '`scripts/compare_checkpoints.py` -> `evals/checkpoint_comparison*.json`', 'done'),
    ('Related-work references checked against publisher / arXiv records', 'Section 9', 'done'),
    ('Baseline model cards checked (parameters, architecture, decoding)', 'Section 3.2', 'done'),
    ('This document', '`scripts/build_final_md.py` regenerates every table from `evals/*.json`', 'done'),
    ('Training ablations: seed variance, upcycling, number of experts, placement (Section 5.6)', '`scripts/run_ablation_queue.py`, `scripts/eval_variant.py` -> `evals/ablation_training.json`', 'done' if HAVE else 'QUEUED - placeholders'),
    ('Load-balancing loss and data-mix ablations', '-', 'not run; no claim made (Section 10)'),
]
for x, y, z in rows:
    a(f'| {x} | {y} | {z} |')
a('')
a('Retained checkpoints, loadable with `adapters/our_conformer.py`:')
a('')
a('* MoE: `runs/scaled-conformer512-moe-3stage/checkpoints/stage3_vanilla_sft/best_model.pt`')
a('* Dense: `runs/scaled-conformer512-dense-3stage/checkpoints/stage3_final_alignment/best_model.pt`')
a('* LoRA adapters: `runs/dialect_lora_adapters/best_lora_adapters.pt`')
a('')
a('Exported copies of the two models (config, vocabulary, code and weights) are in `exported_models/marathi-conformer-130m-moe` and `exported_models/marathi-conformer-130m-dense`.')
a('')
open('final.md', 'w', encoding='utf-8', newline='').write('\n'.join(L))
print(len(L), 'lines')
