import json

with open("scratch_samples.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dialect_names = {
    "D1": "D1: Malvani / Konkan",
    "D2": "D2: Ahirani / Khandesh",
    "D3": "D3: Standard Marathi",
    "D4": "D4: Varhadi / Vidarbha"
}

md = "# Extended Dialect Qualitative Analysis (KenLM alpha=0.15)\n\n"
md += "This comprehensive report highlights the Best, Moderate, and Worst predictions for each dialect across the RESPIN test set when decoded with a 5-gram KenLM constraint.\n\n"

for d_code, d_name in dialect_names.items():
    md += f"## {d_name}\n\n"
    
    samples = data.get(d_code, {})
    
    # BEST
    md += "### Best Samples (Flawless Predictions)\n\n"
    for i, s in enumerate(samples.get("best", [])):
        md += f"**Sample {i+1}** (CER: {s['our_cer']}%)\n"
        md += f"- **Reference**: {s['reference']}\n"
        md += f"- **Our Model**: {s['our_model_pred']}\n"
        md += f"- **Indic Conformer**: {s['indic_pred']}\n\n"
        
    # MODERATE
    md += "### Moderately Good Samples (Minor phonetic or punctuation errors)\n\n"
    for i, s in enumerate(samples.get("moderate", [])):
        md += f"**Sample {i+1}** (CER: {s['our_cer']}%)\n"
        md += f"- **Reference**: {s['reference']}\n"
        md += f"- **Our Model**: {s['our_model_pred']}\n"
        md += f"- **Indic Conformer**: {s['indic_pred']}\n\n"
        
    # WORST
    md += "### Worst Samples (Catastrophic Failures & LM Erasure)\n\n"
    for i, s in enumerate(samples.get("worst", [])):
        md += f"**Sample {i+1}** (CER: {s['our_cer']}%)\n"
        md += f"- **Reference**: {s['reference']}\n"
        md += f"- **Our Model**: {s['our_model_pred']}\n"
        md += f"- **Indic Conformer**: {s['indic_pred']}\n\n"
        
    md += "---\n\n"

with open("inference/evaluation2_kenlm/qualitative_analysis_extended.md", "w", encoding="utf-8") as f:
    f.write(md)

print("Extended report generated!")
