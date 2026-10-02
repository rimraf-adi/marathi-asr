# IISc RESPIN Marathi ASR Benchmark: Model Comparison Report

- **Evaluated Test Set**: IISc RESPIN Held-Out Marathi (`meta_test_mr.json`)
- **Total Utterances**: 2,170 (3.04 hours)
- **Evaluation Protocol**: Zero-leakage held-out test evaluation using **CTC Decoding**
- **Target Dialects**: D1 (Malvani/Konkan), D2 (Ahirani/Khandesh), D3 (Standard Marathi), D4 (Varhadi/Vidarbha)

## 1. Overall System Performance Summary (Aggregate ALL)

| Model Name | Decoding | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our Model (MoE Conformer)** | CTC | **10.78%** | **41.72%** | 91.71% | 8.29% | 0.0047x | 23.8 ms |
| **SraVaani 1.0 (ARTPARK-IISc)** | CTC | **2.97%** | **15.75%** | 58.80% | 41.20% | 0.0159x | 80.4 ms |
| **Indic Conformer 600M (AI4Bharat)** | CTC | **4.95%** | **23.94%** | 70.00% | 30.00% | 0.0643x | 324.3 ms |

## 2. Dialect-by-Dialect Comparative Breakdown

### Character Error Rate (CER %) by Dialect

| Dialect | Name | **Our Model (MoE Conformer)** | **SraVaani 1.0 (ARTPARK-IISc)** | **Indic Conformer 600M (AI4Bharat)** |
| :--- | :--- | :---: | :---: | :---: |
| D1 | Malvani / Konkan | 12.71% | 4.53% | 7.39% |
| D2 | Ahirani / Khandesh | 10.74% | 2.95% | 4.96% |
| D3 | Standard Marathi | 8.04% | 1.69% | 2.27% |
| D4 | Varhadi / Vidarbha | 11.46% | 2.58% | 4.96% |
| ALL | Aggregate | 10.78% | 2.97% | 4.95% |

### Word Error Rate (WER %) by Dialect

| Dialect | Name | **Our Model (MoE Conformer)** | **SraVaani 1.0 (ARTPARK-IISc)** | **Indic Conformer 600M (AI4Bharat)** |
| :--- | :--- | :---: | :---: | :---: |
| D1 | Malvani / Konkan | 48.55% | 21.71% | 34.59% |
| D2 | Ahirani / Khandesh | 43.28% | 17.76% | 24.64% |
| D3 | Standard Marathi | 32.11% | 8.97% | 11.57% |
| D4 | Varhadi / Vidarbha | 42.43% | 14.14% | 24.29% |
| ALL | Aggregate | 41.72% | 15.75% | 23.94% |

## 3. Qualitative Comparative Samples

| Dialect | Reference | **Our Model (MoE Conformer)** | **SraVaani 1.0 (ARTPARK-IISc)** | **Indic Conformer 600M (AI4Bharat)** |
| :---: | :--- | :--- | :--- | :--- |
| D3 | बकरी किंवा मेंढीपासून खत मिळते का ? | बखरी किंवा मेंढीपासून खत मिळते का ?? | बकरी किंवा मेंढीपासून खत मिळते का | बकरी किंवा मेंढीपासून खत मिळते का |
| D3 | ठिबक सिंचनाची जोडणी कशी असावी ? | ठिबक सिंचनाशी जोडणी कशी असावी ?? | ठिबक सिंचनाची जोडणी कशी असावी | ठिबक सिंचनाची जोडणी कशी असावी |
| D3 | के.वाय.सी फॉर्म भरणे गरजेचे असते का ? | के.वाय.सी फॉर्म भरने गरजेचे असते का ?? | केवायसी फॉर्म भरणे गरजेचे असते का | के वाय सी फॉर्म भरणे गरजेचे असते का |
| D3 | मुदत खाते ऑनलाइन उघडता येईल का ? | मुदत खाते ऑनलाइन उघडता येईल का ?? | मुदत खाते ऑनलाइन उघडता येईल का | मुदत खाते ऑनलाइन उघडता येईल का |
| D3 | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या ? | लवकर तयार होणा्या तुरीच्या जाती कोणत्या ?? | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या | लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या |
| D3 | मला एका वर्षातच कर्ज परतफेड करण्याठी दर महिन्याला किमान किती हप्ता बसेल ? | मला एका वर्षातच कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हप्ता बसेल ?? | मला एका वर्षात कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हफ्ता बसेल | मला एका वर्षातच कर्ज परतफेड करण्यासाठी दर महिन्याला किमान किती हप्ता बसेल |
| D3 | कोणत्या वातावरणात कोणते पीक घेऊ नये ? | कोणत्या वातावरणात कोणते पीक घेऊ नये ?? | कोणत्या वातावरणात कोणते पीक घेऊ नये | कोणत्या वातावरणात कोणते पीक घेऊ नये |
| D3 | जमिनीचे व्यवस्थापन कसे करावे ? | जमिनीचे व्यवस्थापन कसे करावे ?? | जमिनीचे व्यवस्थापन कसे करावे | जमिनीचे व्यवस्थापन कसे करावे |
| D3 | खरीप हंगामाचा कोणता कालावधी असतो ? | खरीप हंगामाचा कोणता कालावधी असतो ?? | खरीप हंगामाचा कोणता कालावधी असतो | खरीप हंगामाचा कोणता कालावधी असतो |
| D3 | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ? | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का ?? | पिकांची औषध सरकारी योजनेअंतर्गत मिळणार का | पिकांची औषधे सरकारी योजनेअंतर्गत मिळणार का |
