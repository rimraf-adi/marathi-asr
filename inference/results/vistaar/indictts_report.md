# Vistaar Benchmark Report: `INDICTTS` (Marathi)

- **Total Utterances**: 100 (0.20 hours)
- **Manifest**: `data\vistaar_benchmarks\indictts\marathi\manifest.json`

| Model / Decoding Variant | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our Model (Greedy CTC)** | **12.62%** | **53.00%** | 98.00% | 2.00% | 0.0178x | 129.1 ms |
| **Our Model (Lexicon Beam)** | **12.25%** | **51.60%** | 98.00% | 2.00% | 0.0178x | 129.1 ms |
| **Our Model (KenLM Default α=0.5)** | **13.08%** | **48.50%** | 99.00% | 1.00% | 0.0178x | 129.1 ms |
| **Our Model (KenLM Soft α=0.15)** | **12.09%** | **50.90%** | 94.00% | 6.00% | 0.0178x | 129.1 ms |
| **SraVaani 1.0 (ARTPARK-IISc)** | **3.36%** | **14.10%** | 63.00% | 37.00% | 0.0809x | 586.9 ms |
| **Indic Conformer 600M (AI4Bharat)** | **2.84%** | **13.40%** | 58.00% | 42.00% | 0.0689x | 499.5 ms |
