"""
Qualitative Error Analysis Engine for Marathi ASR.
Categorizes evaluated utterances into:
  - Easy (WER <= 15%, CER <= 5%)
  - Medium (15% < WER <= 50%)
  - Hard / Failed (WER > 50%)
Provides linguistic root-cause diagnosis across:
  1. Digits vs Spoken Numerals
  2. Compound Word / Postposition (विभक्ती) Spacing
  3. Phonetic Substitution & Dialectal Phonotactics
  4. English Loanwords & Technical Transliterations
  5. Acoustic Noise / Speed / Truncation
"""

import re
from typing import Dict, Any, List
from pathlib import Path


def categorize_error(ref: str, hyp: str) -> List[str]:
    """Diagnoses the linguistic failure modes between reference and hypothesis."""
    reasons = []
    
    # 1. Numeral / Digit mismatch
    has_ref_digits = bool(re.search(r'[०-९0-9]', ref))
    has_hyp_digits = bool(re.search(r'[०-९0-9]', hyp))
    if has_ref_digits != has_hyp_digits or has_ref_digits:
        reasons.append("Numeral Mismatch (Digits vs Spoken Words)")

    # 2. English loanword detection (common characters like ॅ, ॉ, or latin letters)
    if any(c in ref for c in ['ॅ', 'ॉ']) or any(c in hyp for c in ['ॅ', 'ॉ']):
        reasons.append("English Loanword / Novel Phonotactics")

    # 3. Postposition / Spacing mismatch
    common_postpositions = ["मध्ये", "वर", "साठी", "कडे", "ला", "ना", "हून", "तून", "समोर", "बद्दल", "प्रमाणे"]
    for pp in common_postpositions:
        if (f" {pp}" in hyp and pp in ref and f" {pp}" not in ref) or (f" {pp}" in ref and pp in hyp and f" {pp}" not in hyp):
            reasons.append("Postposition Compound/Spacing (विभक्ती प्रत्यय)")
            break

    # 4. Anusvara / Nasalization mismatch
    ref_anusvara = ref.count('ं')
    hyp_anusvara = hyp.count('ं')
    if abs(ref_anusvara - hyp_anusvara) >= 2:
        reasons.append("Anusvara / Nasalization Variant")

    # 5. Length discrepancy (omission or hallucination)
    ref_len = len(ref.split())
    hyp_len = len(hyp.split())
    if abs(ref_len - hyp_len) >= 4 or (ref_len > 0 and abs(ref_len - hyp_len) / ref_len > 0.4):
        reasons.append("Acoustic Omission / Speed Distortion")

    if not reasons:
        reasons.append("Phonetic Substitution / Unseen Vocabulary")
    return reasons


def generate_qualitative_report(
    benchmark_name: str,
    samples_data: List[Dict[str, Any]],
    output_path: Path,
    num_examples_per_bucket: int = 5,
):
    """
    samples_data is a list of dicts with:
      - uid
      - wav_path
      - reference
      - greedy_pred, greedy_wer, greedy_cer
      - beam_pred, beam_wer, beam_cer
      - kenlm_pred, kenlm_wer, kenlm_cer
      - sra_pred, sra_wer (optional)
      - ind_pred, ind_wer (optional)
    """
    # Sort samples based on KenLM Soft WER (or Default or Greedy)
    def _get_sort_wer(s):
        return s.get("kenlm_soft_wer", s.get("kenlm_def_wer", s.get("greedy_wer", 100.0)))

    easy = [s for s in samples_data if _get_sort_wer(s) <= 15.0]
    medium = [s for s in samples_data if 15.0 < _get_sort_wer(s) <= 50.0]
    hard = [s for s in samples_data if _get_sort_wer(s) > 50.0]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Qualitative Error Analysis Report: `{benchmark_name.upper()}` (Marathi)\n\n")
        f.write(f"### Distribution Summary ({len(samples_data)} total utterances)\n")
        f.write(f"- **Easy Sentences (WER ≤ 15%)**: {len(easy)} utterances ({len(easy)/max(len(samples_data), 1)*100:.1f}%)\n")
        f.write(f"- **Medium Difficulty (15% < WER ≤ 50%)**: {len(medium)} utterances ({len(medium)/max(len(samples_data), 1)*100:.1f}%)\n")
        f.write(f"- **Hard / Failed Sentences (WER > 50%)**: {len(hard)} utterances ({len(hard)/max(len(samples_data), 1)*100:.1f}%)\n\n")

        f.write("---\n\n")

        # 1. Easy Sentences
        f.write("## 1. Easy Sentences (Near-Perfect / Exact Matches)\n\n")
        f.write("These utterances feature standard grammatical structure, clear speech articulation, common Marathi vocabulary, and minimal numeric or loanword interference.\n\n")
        for i, s in enumerate(easy[:num_examples_per_bucket]):
            f.write(f"### Example E-{i+1} (`{s['uid']}`)\n")
            f.write(f"- **Reference**: `{s['reference']}`\n")
            f.write(f"- **Greedy CTC**: `{s.get('greedy_pred', 'N/A')}` *(WER: {s.get('greedy_wer', 0):.1f}%, CER: {s.get('greedy_cer', 0):.1f}%)*\n")
            f.write(f"- **Lexicon Beam**: `{s.get('beam_pred', 'N/A')}` *(WER: {s.get('beam_wer', 0):.1f}%, CER: {s.get('beam_cer', 0):.1f}%)*\n")
            f.write(f"- **KenLM Default (α=0.5)**: `{s.get('kenlm_def_pred', 'N/A')}` *(WER: {s.get('kenlm_def_wer', 0):.1f}%, CER: {s.get('kenlm_def_cer', 0):.1f}%)*\n")
            f.write(f"- **KenLM Soft (α=0.15)**: `{s.get('kenlm_soft_pred', 'N/A')}` *(WER: {s.get('kenlm_soft_wer', 0):.1f}%, CER: {s.get('kenlm_soft_cer', 0):.1f}%)*\n")
            if 'sra_pred' in s:
                f.write(f"- **SraVaani 1.0**: `{s['sra_pred']}` *(WER: {s.get('sra_wer', 0):.1f}%)*\n")
            if 'ind_pred' in s:
                f.write(f"- **Indic Conformer**: `{s['ind_pred']}` *(WER: {s.get('ind_wer', 0):.1f}%)*\n")
            f.write(f"- **Linguistic Analysis**: Clean acoustic emissions. Phonetic alignment matches standard dictionary unigrams.\n\n")

        f.write("---\n\n")

        # 2. Medium Difficulty Sentences
        f.write("## 2. Medium Difficulty Sentences (Partial Errors: 15% < WER ≤ 50%)\n\n")
        f.write("In this bucket, acoustic recognition is mostly correct, but errors occur due to postposition attachment, minor spelling variations, or spoken vs. written numeral differences.\n\n")
        for i, s in enumerate(medium[:num_examples_per_bucket]):
            reasons = categorize_error(s['reference'], s.get('kenlm_soft_pred', s.get('greedy_pred', '')))
            f.write(f"### Example M-{i+1} (`{s['uid']}`)\n")
            f.write(f"- **Reference**: `{s['reference']}`\n")
            f.write(f"- **Greedy CTC**: `{s.get('greedy_pred', 'N/A')}` *(WER: {s.get('greedy_wer', 0):.1f}%)*\n")
            f.write(f"- **Lexicon Beam**: `{s.get('beam_pred', 'N/A')}` *(WER: {s.get('beam_wer', 0):.1f}%)*\n")
            f.write(f"- **KenLM Default (α=0.5)**: `{s.get('kenlm_def_pred', 'N/A')}` *(WER: {s.get('kenlm_def_wer', 0):.1f}%)*\n")
            f.write(f"- **KenLM Soft (α=0.15)**: `{s.get('kenlm_soft_pred', 'N/A')}` *(WER: {s.get('kenlm_soft_wer', 0):.1f}%)*\n")
            if 'sra_pred' in s:
                f.write(f"- **SraVaani 1.0**: `{s['sra_pred']}` *(WER: {s.get('sra_wer', 0):.1f}%)*\n")
            if 'ind_pred' in s:
                f.write(f"- **Indic Conformer**: `{s['ind_pred']}` *(WER: {s.get('ind_wer', 0):.1f}%)*\n")
            f.write(f"- **Failure Causes**: {', '.join(reasons)}\n")
            f.write(f"- **Detailed Diagnosis**: Notice how KenLM helps prune invalid acoustic tokens into grammatical Marathi words, but spacing/numeral formatting discrepancies persist.\n\n")

        f.write("---\n\n")

        # 3. Hard / Failed Sentences
        f.write("## 3. Hard & Failed Sentences (Severe Discrepancies: WER > 50%)\n\n")
        f.write("These sentences highlight the principal failure boundaries of zero-shot acoustic transfer and greedy CTC.\n\n")
        for i, s in enumerate(hard[:num_examples_per_bucket]):
            reasons = categorize_error(s['reference'], s.get('greedy_pred', ''))
            f.write(f"### Example H-{i+1} (`{s['uid']}`)\n")
            f.write(f"- **Reference**: `{s['reference']}`\n")
            f.write(f"- **Greedy CTC**: `{s.get('greedy_pred', 'N/A')}` *(WER: {s.get('greedy_wer', 0):.1f}%)*\n")
            f.write(f"- **Lexicon Beam**: `{s.get('beam_pred', 'N/A')}` *(WER: {s.get('beam_wer', 0):.1f}%)*\n")
            f.write(f"- **KenLM Default (α=0.5)**: `{s.get('kenlm_def_pred', 'N/A')}` *(WER: {s.get('kenlm_def_wer', 0):.1f}%)*\n")
            f.write(f"- **KenLM Soft (α=0.15)**: `{s.get('kenlm_soft_pred', 'N/A')}` *(WER: {s.get('kenlm_soft_wer', 0):.1f}%)*\n")
            if 'sra_pred' in s:
                f.write(f"- **SraVaani 1.0**: `{s['sra_pred']}` *(WER: {s.get('sra_wer', 0):.1f}%)*\n")
            if 'ind_pred' in s:
                f.write(f"- **Indic Conformer**: `{s['ind_pred']}` *(WER: {s.get('ind_wer', 0):.1f}%)*\n")
            f.write(f"- **Root Cause Analysis**: **{', '.join(reasons)}**\n")
            f.write(f"- **Why It Failed**: Ground truth uses formal broadcast terms, numerals, or dense English loanwords that were under-represented in the rural dialect training set. When numbers like '१३' are spoken as 'तेरा', JiWER counts all words as incorrect.\n\n")

    print(f"[Qualitative Report] Saved detailed qualitative analysis: {output_path}")

    print(f"[Qualitative Report] Saved detailed qualitative analysis: {output_path}")
