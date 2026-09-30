# Vistaar Benchmark Report: `KATHBATH_NOISY` (Marathi)

- **Total Utterances**: 1631 (3.05 hours)
- **Manifest**: `data\vistaar_benchmarks\kathbath_noisy\marathi\manifest.json`

| Model / Decoding Variant | CER (%) | WER (%) | SER (%) | Exact Match (%) | RTF | Latency (ms/utt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our Model (Greedy CTC)** | **25.62%** | **69.95%** | 99.94% | 0.06% | 0.0225x | 151.5 ms |
| **Our Model (Lexicon Beam)** | **25.29%** | **69.68%** | 99.94% | 0.06% | 0.0225x | 151.5 ms |
| **Our Model (KenLM Default α=0.5)** | **25.84%** | **61.71%** | 99.51% | 0.49% | 0.0225x | 151.5 ms |
| **Our Model (KenLM Soft α=0.15)** | **24.87%** | **69.56%** | 99.45% | 0.55% | 0.0225x | 151.5 ms |
| **SraVaani 1.0 (ARTPARK-IISc)** | **6.29%** | **18.98%** | 78.60% | 21.40% | 0.0001x | 0.6 ms |
| **Indic Conformer 600M (AI4Bharat)** | **6.15%** | **18.86%** | 78.11% | 21.89% | 0.0631x | 425.3 ms |
