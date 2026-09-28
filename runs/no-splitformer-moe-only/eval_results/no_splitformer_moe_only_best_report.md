# Benchmark Report: `no_splitformer_moe_only_best`

- **Checkpoint**: `runs/no-splitformer-moe-only/checkpoints/stage3_final_alignment/best_model.pt`
- **Streaming Chunk Size**: 16 frames (640 ms)
- **Evaluated Test Set**: IISc RESPIN Held-Out (2,170 utterances, 3.04 hrs)

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
| Layer 12 (Deep) | D1 (Malvani / Konkan) | **11.40%** | 43.04% | 559 | 0.0019x |
| Layer 12 (Deep) | D2 (Ahirani / Khandesh) | **8.53%** | 33.84% | 540 | 0.0019x |
| Layer 12 (Deep) | D3 (Standard Marathi) | **5.63%** | 20.89% | 555 | 0.0019x |
| Layer 12 (Deep) | D4 (Varhadi / Vidarbha) | **7.93%** | 30.45% | 516 | 0.0019x |
| Layer 12 (Deep) | ALL (Aggregate) | **8.43%** | 32.21% | 2170 | 0.0019x |

## Qualitative Prediction Samples (Layer 12)

| Dialect | Reference | Prediction | CER (%) | WER (%) |
| :--- | :--- | :--- | :--- | :--- |
| D3 | बकरी किंवा मेंढीपासून खत मिळते का ? | बकरी किंवा मेंढी पासून खत मिळते का ?? | 5.71% | 42.86% |
| D3 | ठिबक सिंचनाची जोडणी कशी असावी ? | ठिबक सिंचनाची जोडणी कशी असावी ?? | 3.23% | 16.67% |
| D3 | के.वाय.सी फॉर्म भरणे गरजेचे असते का ? | के.वाय.सी फॉर्म भरणे गरजेचे असते का ?? | 2.7% | 14.29% |
| D3 | मुदत खाते ऑनलाइन उघडता येईल का ? | मुदत खाते ऑनलाइन उघडता येईल का ?? | 3.12% | 14.29% |
| D3 | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या ? | लवकर तयार होणार्या तुरीच्या जाती कोणत्या ?? | 4.76% | 28.57% |
| D3 | मला एका वर्षातच कर्ज परतफेड करण्याठी दर महिन्याला किमान किती हप्ता बसेल ? | मला एका वर्षातच कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हप्ता बसेल ? | 2.74% | 7.69% |
| D3 | कोणत्या वातावरणात कोणते पीक घेऊ नये ? | कोणत्या वातावरणात कोणते पीक घेऊ नये ?? | 2.7% | 14.29% |
| D3 | जमिनीचे व्यवस्थापन कसे करावे ? | जमिनीचे व्यवस्थापन कसे करावे | 6.67% | 20.0% |
| D3 | खरीप हंगामाचा कोणता कालावधी असतो ? | खरीप हंगामाचा कोणता कालावधी असतो ?? | 2.94% | 16.67% |
| D3 | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ? | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ?? | 2.27% | 14.29% |
