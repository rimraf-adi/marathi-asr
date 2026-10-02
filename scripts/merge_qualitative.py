import pandas as pd
import os

print("Loading detailed CSVs...")
df_our = pd.read_csv("inference/evaluation2/our_model_respin_detailed.csv")
df_sravaani = pd.read_csv("inference/evaluation2/sravaani_respin_detailed.csv")
df_indic = pd.read_csv("inference/evaluation2/indic_conformer_respin_detailed.csv")

df_our = df_our[['uid', 'dialect', 'domain', 'duration_sec', 'reference', 'prediction', 'norm_cer_percent', 'norm_wer_percent']].rename(
    columns={'prediction': 'our_model_pred', 'norm_cer_percent': 'our_cer', 'norm_wer_percent': 'our_wer'}
)

df_sravaani = df_sravaani[['uid', 'prediction', 'norm_cer_percent', 'norm_wer_percent']].rename(
    columns={'prediction': 'sravaani_pred', 'norm_cer_percent': 'sravaani_cer', 'norm_wer_percent': 'sravaani_wer'}
)

df_indic = df_indic[['uid', 'prediction', 'norm_cer_percent', 'norm_wer_percent']].rename(
    columns={'prediction': 'indic_pred', 'norm_cer_percent': 'indic_cer', 'norm_wer_percent': 'indic_wer'}
)

print("Merging dataframes...")
merged = pd.merge(df_our, df_sravaani, on="uid", how="left")
merged = pd.merge(merged, df_indic, on="uid", how="left")

out_path = "inference/evaluation2/all_models_qualitative_analysis.csv"
merged.to_csv(out_path, index=False)
print(f"Saved master qualitative analysis CSV to: {out_path}")
