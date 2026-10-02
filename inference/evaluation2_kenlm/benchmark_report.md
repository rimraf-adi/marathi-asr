# IISc RESPIN Marathi ASR Benchmark: Model Comparison Report

- **Evaluated Test Set**: IISc RESPIN Held-Out Marathi (`meta_test_mr.json`)
- **Total Utterances**: 2,170 (3.04 hours)
- **Evaluation Protocol**: Zero-leakage held-out test evaluation using **CTC Decoding**
- **Target Dialects**: D1 (Malvani/Konkan), D2 (Ahirani/Khandesh), D3 (Standard Marathi), D4 (Varhadi/Vidarbha)

## 1. Overall System Performance Summary (Aggregate ALL)

| Model Name | Decoding | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our Model (MoE Conformer)** | CTC | **8.33%** | **32.60%** | 81.98% | 18.02% | 0.0063x | 31.6 ms |
| **SraVaani 1.0 (ARTPARK-IISc)** | CTC | **2.97%** | **15.75%** | 58.80% | 41.20% | 0.0148x | 74.6 ms |
| **Indic Conformer 600M (AI4Bharat)** | CTC | **4.95%** | **23.94%** | 70.00% | 30.00% | 0.0632x | 319.1 ms |

## 2. Dialect-by-Dialect Comparative Breakdown

### Character Error Rate (CER %) by Dialect

| Dialect | Name | **Our Model (MoE Conformer)** | **SraVaani 1.0 (ARTPARK-IISc)** | **Indic Conformer 600M (AI4Bharat)** |
| :--- | :--- | :---: | :---: | :---: |
| D1 | Malvani / Konkan | 11.09% | 4.53% | 7.39% |
| D2 | Ahirani / Khandesh | 8.27% | 2.95% | 4.96% |
| D3 | Standard Marathi | 5.14% | 1.69% | 2.27% |
| D4 | Varhadi / Vidarbha | 8.57% | 2.58% | 4.96% |
| ALL | Aggregate | 8.33% | 2.97% | 4.95% |

### Word Error Rate (WER %) by Dialect

| Dialect | Name | **Our Model (MoE Conformer)** | **SraVaani 1.0 (ARTPARK-IISc)** | **Indic Conformer 600M (AI4Bharat)** |
| :--- | :--- | :---: | :---: | :---: |
| D1 | Malvani / Konkan | 43.46% | 21.71% | 34.59% |
| D2 | Ahirani / Khandesh | 33.81% | 17.76% | 24.64% |
| D3 | Standard Marathi | 19.80% | 8.97% | 11.57% |
| D4 | Varhadi / Vidarbha | 32.62% | 14.14% | 24.29% |
| ALL | Aggregate | 32.60% | 15.75% | 23.94% |

## 3. Qualitative Comparative Samples

| Dialect | Reference | **Our Model (MoE Conformer)** | **SraVaani 1.0 (ARTPARK-IISc)** | **Indic Conformer 600M (AI4Bharat)** |
| :---: | :--- | :--- | :--- | :--- |
| D3 | बकरी किंवा मेंढीपासून खत मिळते का ? | बकरी किंवा मेंढी पासून खत मिळते का ?? | बकरी किंवा मेंढीपासून खत मिळते का | बकरी किंवा मेंढीपासून खत मिळते का |
| D3 | ठिबक सिंचनाची जोडणी कशी असावी ? | ठिबक सिंचनाची जोडणी कशी असावी ?? | ठिबक सिंचनाची जोडणी कशी असावी | ठिबक सिंचनाची जोडणी कशी असावी |
| D3 | के.वाय.सी फॉर्म भरणे गरजेचे असते का ? | के.वाय.सी फॉर्म भरणे गरजेचे असते का ?? | केवायसी फॉर्म भरणे गरजेचे असते का | के वाय सी फॉर्म भरणे गरजेचे असते का |
| D3 | मुदत खाते ऑनलाइन उघडता येईल का ? | मुदत खाते ऑनलाइन उघडता येईल का ?? | मुदत खाते ऑनलाइन उघडता येईल का | मुदत खाते ऑनलाइन उघडता येईल का |
| D3 | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या ? | लवकर तयार होणार्या तुरीच्या जाती कोणत्या ?? | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या |
| D3 | मला एका वर्षातच कर्ज परतफेड करण्याठी दर महिन्याला किमान किती हप्ता बसेल ? | मला एका वर्षातच कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हप्ता बसेल | मला एका वर्षात कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हफ्ता बसेल | मला एका वर्षातच कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हप्ता बसेल |
| D3 | कोणत्या वातावरणात कोणते पीक घेऊ नये ? | कोणत्या वातावरणात कोणते पीक घेऊ नये ?? | कोणत्या वातावरणात कोणते पीक घेऊ नये | कोणत्या वातावरणात कोणते पीक घेऊ नये |
| D3 | जमिनीचे व्यवस्थापन कसे करावे ? | जमिनीचे व्यवस्थापन कसे करावे ?? | जमिनीचे व्यवस्थापन कसे करावे | जमिनीचे व्यवस्थापन कसे करावे |
| D3 | खरीप हंगामाचा कोणता कालावधी असतो ? | खरीप हंगामाचा कोणता कालावधी असतो ?? | खरीप हंगामाचा कोणता कालावधी असतो | खरीप हंगामाचा कोणता कालावधी असतो |
| D3 | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ? | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ?? | पिकांची औषध सरकारी योजनेअंतर्गत मिळणार का | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का |
