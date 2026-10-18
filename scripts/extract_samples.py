import pandas as pd
import json

df = pd.read_csv("inference/evaluation2_kenlm/all_models_qualitative_analysis.csv")

dialects = ["D1", "D2", "D3", "D4"]
results = {}

for d in dialects:
    df_d = df[df['dialect'] == d]
    
    # Best: CER == 0 or lowest
    best_df = df_d.sort_values('our_cer').head(3)
    
    # Moderately good: CER between 5 and 20 (not perfect, but good)
    good_df = df_d[(df_d['our_cer'] > 0) & (df_d['our_cer'] < 15)].sort_values('our_cer').head(10)
    if len(good_df) >= 3:
        good_df = good_df.sample(3)
    else:
        good_df = df_d.sort_values('our_cer').iloc[3:6]
        
    # Worst of the worst: Highest CER
    bad_df = df_d.sort_values('our_cer', ascending=False).head(3)
    
    results[d] = {
        "best": best_df[['reference', 'our_model_pred', 'indic_pred', 'our_cer']].to_dict('records'),
        "moderate": good_df[['reference', 'our_model_pred', 'indic_pred', 'our_cer']].to_dict('records'),
        "worst": bad_df[['reference', 'our_model_pred', 'indic_pred', 'our_cer']].to_dict('records')
    }

with open("scratch_samples.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Samples extracted to scratch_samples.json")
