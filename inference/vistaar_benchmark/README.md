# Vistaar Benchmark Suite for Marathi ASR

Based on the [AI4Bharat/Vistaar](https://github.com/AI4Bharat/Vistaar) repository.

Vistaar is a comprehensive benchmark comprising diverse public datasets across multiple domains (news, education, literature, tourism, noisy speech).

---

## 1. Marathi Benchmark Subsets in Vistaar

The Marathi (`mr`) subset spans 6 diverse benchmarks:

| Benchmark Dataset | Domain / Condition | IndicWhisper Baseline WER (%) | ObjectStore Download URL |
| :--- | :--- | :---: | :--- |
| **Kathbath** | Crowdsourced read speech | **19.9%** | `https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath.zip` |
| **Kathbath Hard** | Acoustic noise / reverberation | **22.1%** | `https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath_noisy.zip` |
| **CommonVoice** | Crowdsourced multi-speaker | **22.8%** | `https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/commonvoice.zip` |
| **FLEURS** | Conversational / read speech | **20.5%** | `https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/fleurs.zip` |
| **IndicTTS** | Synthetic read speech | **11.4%** | `https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/indictts.zip` |
| **MUCS** | Multilingual domain speech | **12.8%** | `https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/mucs.zip` |
| **Average (Marathi)** | **Aggregate across all 6** | **18.2%** | — |

---

## 2. IndicWhisper Marathi Model Checkpoint

The official fine-tuned IndicWhisper checkpoint for Marathi can be acquired from:
- `https://indicwhisper.objectstore.e2enetworks.net/marathi_models.zip`

---

## 3. Manifest Specification

Vistaar uses JSON Lines (`.json` or `.jsonl`) manifests formatted as:
```json
{"audio_filepath": "/path/to/marathi/audio1.wav", "duration": 4.52, "text": "मराठीतील वाक्य येथे येईल"}
{"audio_filepath": "/path/to/marathi/audio2.wav", "duration": 5.10, "text": "दुसरे वाक्य"}
```

Directory layout after extraction:
```
data/vistaar_benchmarks/
├── kathbath/
│   └── marathi/
│       ├── audio/
│       └── transcript.txt
├── kathbath_noisy/
│   └── marathi/
│       ├── audio/
│       └── transcript.txt
├── commonvoice/
│   └── marathi/
├── fleurs/
│   └── marathi/
├── indictts/
│   └── marathi/
└── mucs/
    └── marathi/
```

---

## 4. Planned Evaluation Workflow

```bash
# 1. Download and prepare Marathi slices:
python inference/vistaar_benchmark/prepare_vistaar.py --datasets kathbath,fleurs,commonvoice

# 2. Run full evaluation across all models (Our Model, IndicWhisper, SraVaani, IndicConformer):
python inference/vistaar_benchmark/eval_vistaar.py --model all --benchmark kathbath
```
