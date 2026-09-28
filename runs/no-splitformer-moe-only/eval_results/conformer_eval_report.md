# Benchmark Report: `conformer_eval`

- **Checkpoint**: `runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt`
- **Streaming Chunk Size**: 16 frames (640 ms)
- **Evaluated Test Set**: IISc RESPIN Held-Out (2 utterances, 0.00 hrs)

| Exit Level | Dialect | CER (%) | WER (%) | Utterances | RTF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Layer 4 (Fast) | D1 (Malvani / Konkan) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 4 (Fast) | D2 (Ahirani / Khandesh) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 4 (Fast) | D3 (Standard Marathi) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 4 (Fast) | D4 (Varhadi / Vidarbha) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 4 (Fast) | ALL (Aggregate) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 8 (Balanced) | D1 (Malvani / Konkan) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 8 (Balanced) | D2 (Ahirani / Khandesh) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 8 (Balanced) | D3 (Standard Marathi) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 8 (Balanced) | D4 (Varhadi / Vidarbha) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 8 (Balanced) | ALL (Aggregate) | **0.00%** | 0.00% | 0 | 0.0x |
| Layer 12 (Deep) | D1 (Malvani / Konkan) | **0.00%** | 0.00% | 0 | 0.0796x |
| Layer 12 (Deep) | D2 (Ahirani / Khandesh) | **0.00%** | 0.00% | 0 | 0.0796x |
| Layer 12 (Deep) | D3 (Standard Marathi) | **1.52%** | 15.38% | 2 | 0.0796x |
| Layer 12 (Deep) | D4 (Varhadi / Vidarbha) | **0.00%** | 0.00% | 0 | 0.0796x |
| Layer 12 (Deep) | ALL (Aggregate) | **1.52%** | 15.38% | 2 | 0.0796x |

## Qualitative Prediction Samples (Layer 12)

| Dialect | Reference | Prediction | CER (%) | WER (%) |
| :--- | :--- | :--- | :--- | :--- |
| D3 | बकरी किंवा मेंढीपासून खत मिळते का ? | बकरी किंवा मेंढी पासून खत मिळते का ? | 2.86% | 28.57% |
| D3 | ठिबक सिंचनाची जोडणी कशी असावी ? | ठिबक सिंचनाची जोडणी कशी असावी ? | 0.0% | 0.0% |
