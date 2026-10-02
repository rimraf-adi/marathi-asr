import pandas as pd
import json

df = pd.read_csv("inference/evaluation2/all_models_qualitative_analysis.csv")

dialects = df['dialect'].unique()
results = {}

for d in dialects:
    df_d = df[df['dialect'] == d]
    
    # Moderately good: CER between 0 and 15 (not perfect, but good)
    good_df = df_d[(df_d['our_cer'] > 0) & (df_d['our_cer'] < 15)].sort_values('our_cer').head(10)
    # If not enough, just take lowest CER
    if len(good_df) < 3:
        good_df = df_d.sort_values('our_cer').head(3)
    else:
        good_df = good_df.sample(3)
        
    # Worst of the worst: Highest CER
    bad_df = df_d.sort_values('our_cer', ascending=False).head(3)
    
    results[d] = {
        "good": good_df[['reference', 'our_model_pred', 'indic_pred', 'our_cer']].to_dict('records'),
        "bad": bad_df[['reference', 'our_model_pred', 'indic_pred', 'our_cer']].to_dict('records')
    }

with open("scratch_samples.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Samples extracted to scratch_samples.json")
