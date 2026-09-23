# Data Utilities (`data_utils`)

This directory contains standalone diagnostic, inspection, and verification scripts for the Marathi speech datasets used in this project:

| Script | Purpose | Target Dataset |
| :--- | :--- | :--- |
| `check_all_dialects.py` | Audits available Marathi-belt configs and sizes in Vaani | ARTPARK-IISc/Vaani |
| `check_info.py` | Inspects dataset builder info, feature schemas, and splits | ai4bharat/Shrutilipi |
| `check_sizes.py` | Calculates and displays total download and audio GBs across configs | Shrutilipi + Vaani |
| `inspect_datasets.py` | Streams single audio samples from each available Marathi dialect | ARTPARK-IISc/Vaani |
| `load_shrutilipi.py` | Demonstrates on-the-fly streaming and audio inspection | ai4bharat/Shrutilipi |

Production streaming data loaders used for training live in:
- `pretraining/dataset_loader.py` (Interleaved Shrutilipi & Vaani streaming)
- `causal_alignment/dataset.py` (Supervised CTC StreamingCTCPrefetchLoader)
- `moe/respin_dataset.py` (Local IISc RESPIN multi-dialect loader with speed/noise aug)
