# Benchmark Report: `scaled-conformer512-moe-3stage_eval`

- **Checkpoint**: `runs\scaled-conformer512-moe-3stage\checkpoints\stage3_final_alignment\best_model.pt`
- **Streaming Chunk Size**: None frames (Full Context ms)
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
| Layer 12 (Deep) | D1 (Malvani / Konkan) | **9.12%** | 35.35% | 559 | 0.0025x |
| Layer 12 (Deep) | D2 (Ahirani / Khandesh) | **7.16%** | 29.58% | 540 | 0.0025x |
| Layer 12 (Deep) | D3 (Standard Marathi) | **5.78%** | 21.11% | 555 | 0.0025x |
| Layer 12 (Deep) | D4 (Varhadi / Vidarbha) | **6.85%** | 27.26% | 516 | 0.0025x |
| Layer 12 (Deep) | ALL (Aggregate) | **7.26%** | 28.42% | 2170 | 0.0025x |

## Qualitative Prediction Samples (Layer 12)

| Dialect | Reference | Prediction | CER (%) | WER (%) |
| :--- | :--- | :--- | :--- | :--- |
| D3 | बकरी किंवा मेंढीपासून खत मिळते का ? | बकरी किंवा मेंढीपासून खत मिळते का ? | 0.0% | 0.0% |
| D3 | ठिबक सिंचनाची जोडणी कशी असावी ? | ठिबक सिंचनाची जोडणी कशी असावी ? ? | 6.45% | 16.67% |
| D3 | के.वाय.सी फॉर्म भरणे गरजेचे असते का ? | के.वाय.सी फॉर्म भरणे गरजेचे असते का ? | 0.0% | 0.0% |
| D3 | मुदत खाते ऑनलाइन उघडता येईल का ? | मुदत खाते ऑनलाइन उघडता येईल का ? ? | 6.25% | 14.29% |
| D3 | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या ? | लवकर तयार होणा⁇्या तुडीच्या जाती कोणत्या ? ? | 9.52% | 42.86% |
| D3 | मला एका वर्षातच कर्ज परतफेड करण्याठी दर महिन्याला किमान किती हप्ता बसेल ? | मला एका वर्षा तच कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हफ्ता भसेल ? | 6.85% | 38.46% |
| D3 | कोणत्या वातावरणात कोणते पीक घेऊ नये ? | कोणत्या वातावरणात कोणते पीक घेऊ नये ? ? | 5.41% | 14.29% |
| D3 | जमिनीचे व्यवस्थापन कसे करावे ? | जमिनीचे व्यवस्थापन कसे करावे ? ? | 6.67% | 20.0% |
| D3 | खरीप हंगामाचा कोणता कालावधी असतो ? | खरीप हंगामाचा कोणता कालावधी असतो ? ? | 5.88% | 16.67% |
| D3 | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ? | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ? | 0.0% | 0.0% |
