# Comprehensive Hardware & Efficiency Benchmark: Marathi ASR Models

- **Benchmarking Environment**: `Windows-10-10.0.19045-SP0`
- **CPU**: 16 Cores
- **GPU**: `NVIDIA RTX A5000` (23.99 GB VRAM, CUDA 12.1)

## 1. Model Footprint, Active Parameters & Sparsity Ratio

| Model Name | Architecture | Total Params | Active Params | Sparsity Ratio | Checkpoint Disk (MB) | Static VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Our Model (MoE Conformer)** | Conformer + Sparse MoE (Top-2 / 4 Experts) | 32.84M | **26.53M** | 80.8% | 320.3 MB | 160.5 MB |
| **SraVaani 1.0 (ARTPARK-IISc)** | FastConformer + Hybrid TDT-CTC (FP16) | 430.0M | **430.0M** | 100.0% (Dense) | 866.7 MB | 1,796.4 MB |
| **Indic Conformer 600M (AI4Bharat)** | Conformer Dense (600M Multilingual) | 600.0M | **600.0M** | 100.0% (Dense) | 2,400.0 MB | ONNX CPU Engine |


## 2. Latency & Real-Time Factor (RTF) across Audio Durations (Batch Size = 1)

| Model Name | Audio Duration | Mean Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | RTF | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our Model (MoE Conformer)** | 3.0s | 78.85 ms | 77.39 ms | 85.59 ms | 89.5 ms | **0.0263x** | 185.95 MB |
| **Our Model (MoE Conformer)** | 8.0s | 89.32 ms | 88.51 ms | 95.39 ms | 114.78 ms | **0.0112x** | 206.9 MB |
| **Our Model (MoE Conformer)** | 15.0s | 79.4 ms | 81.52 ms | 89.23 ms | 91.46 ms | **0.0053x** | 239.38 MB |
| **SraVaani 1.0 (ARTPARK-IISc)** | 5.15s | 1711.33 ms | 122.66 ms | 1716.2 ms | 14580.29 ms | **0.332x** | 1796.4 MB |
| **Indic Conformer 600M (AI4Bharat)** | 5.15s | 968.35 ms | 787.77 ms | 1379.17 ms | 1705.8 ms | **0.1879x** | 0.0 MB |


## 3. GPU Batch Scaling & Throughput (Hours of Audio / Minute)

| Model Name | Batch Size | Latency (ms) | Throughput (Audio Hrs / Min) | Utterances / Sec | RTF | Peak VRAM (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Our Model (MoE Conformer)** | B=1 | 78.33 ms | **1.06 hrs/min** | 12.77 utt/s | 0.0157x | 193.44 MB | OK |
| **Our Model (MoE Conformer)** | B=4 | 84.33 ms | **3.95 hrs/min** | 47.43 utt/s | 0.0042x | 262.34 MB | OK |
| **Our Model (MoE Conformer)** | B=8 | 88.91 ms | **7.5 hrs/min** | 89.98 utt/s | 0.0022x | 351.45 MB | OK |
| **Our Model (MoE Conformer)** | B=16 | 75.74 ms | **17.6 hrs/min** | 211.25 utt/s | 0.0009x | 533.76 MB | OK |
| **Our Model (MoE Conformer)** | B=32 | 84.78 ms | **31.46 hrs/min** | 377.47 utt/s | 0.0005x | 892.07 MB | OK |
| **Our Model (MoE Conformer)** | B=64 | 115.15 ms | **46.31 hrs/min** | 555.77 utt/s | 0.0004x | 1614.45 MB | OK |
| **SraVaani 1.0 (ARTPARK-IISc)** | B=1 | 119.97 ms | **0.72 hrs/min** | 8.34 utt/s | 0.0233x | 1796.4 MB | OK |
| **SraVaani 1.0 (ARTPARK-IISc)** | B=4 | 5565.61 ms | **0.06 hrs/min** | 0.72 utt/s | 0.2902x | 1852.91 MB | OK |
| **SraVaani 1.0 (ARTPARK-IISc)** | B=8 | 6765.12 ms | **0.11 hrs/min** | 1.18 utt/s | 0.1518x | 1995.05 MB | OK |
| **SraVaani 1.0 (ARTPARK-IISc)** | B=16 | 2811.62 ms | **0.54 hrs/min** | 5.69 utt/s | 0.0311x | 2227.71 MB | OK |
| **SraVaani 1.0 (ARTPARK-IISc)** | B=32 | 2565.27 ms | **1.11 hrs/min** | 12.47 utt/s | 0.015x | 2690.23 MB | OK |


## 4. CPU Thread Scaling (Edge / Serverless without GPU)

| Model Name | CPU Threads | Latency (5s Audio) | RTF |
| :--- | :---: | :---: | :---: |
| **Our Model (MoE Conformer)** | 1 Threads | 441.19 ms | **0.0882x** |
| **Our Model (MoE Conformer)** | 4 Threads | 242.4 ms | **0.0485x** |
| **Our Model (MoE Conformer)** | 8 Threads | 215.61 ms | **0.0431x** |
| **Our Model (MoE Conformer)** | 16 Threads | 261.7 ms | **0.0523x** |
| **Indic Conformer 600M (AI4Bharat)** | 1 Threads | 700.67 ms | **0.1359x** |
| **Indic Conformer 600M (AI4Bharat)** | 4 Threads | 706.87 ms | **0.1371x** |
| **Indic Conformer 600M (AI4Bharat)** | 8 Threads | 714.38 ms | **0.1386x** |
| **Indic Conformer 600M (AI4Bharat)** | 16 Threads | 755.34 ms | **0.1465x** |


## 5. Decoder Overhead Breakdown (Our Model)

| Decoding Strategy | Mean Latency (ms) | P50 (ms) | P90 (ms) |
| :--- | :---: | :---: | :---: |
| **Pure Greedy CTC (argmax)** | 0.02 ms | 0.02 ms | 0.03 ms |
| **Prefix Beam Search (Lexicon only)** | 108.48 ms | 107.46 ms | 109.66 ms |
| **Prefix Beam + KenLM 5-gram Default (α=0.5)** | 78.52 ms | 77.99 ms | 79.77 ms |
| **Prefix Beam + KenLM 5-gram Soft (α=0.15)** | 78.78 ms | 78.43 ms | 79.6 ms |
