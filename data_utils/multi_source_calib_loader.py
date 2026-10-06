"""
Multi-Source Calibration & Final Alignment Data Loader.
Combines 4 distinct speech corpora for Stage 3 Final Calibration & Alignment:
  1. RESPIN 5% Held-Out Dialect Split (~41,041 utterances, D1-D4 regional dialects)
  2. OpenSLR64 Marathi Speech Corpus (High-quality clean crowdsourced speech)
  3. AI4Bharat Kathbath Marathi Corpus (~84,070 utterances of diverse read speech)
  4. Pretraining Corpus (Streaming ai4bharat/Shrutilipi & ARTPARK-IISc/Vaani)
"""

import os
import random
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional, Iterator
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript
from data_utils.kathbath_loader import load_kathbath_manifest, apply_audio_augmentation
from moe.respin_dataset import build_or_load_speaker_split


def collate_multi_source_batch(
    samples: List[Dict[str, Any]],
    tokenizer: MarathiTokenizer,
    augment: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Pads variable-length audio and encodes transcripts for CTCLoss.
    """
    valid = []
    for s in samples:
        try:
            # 1. Audio loading
            if "audio" in s and s["audio"] is not None:
                audio = s["audio"]
                sr = s.get("sampling_rate", 16000)
                if isinstance(audio, np.ndarray):
                    t_audio = torch.from_numpy(audio).float()
                else:
                    t_audio = audio.float()
            elif "wav_path" in s and os.path.exists(s["wav_path"]):
                audio_arr, sr = sf.read(s["wav_path"], dtype="float32")
                if len(audio_arr.shape) > 1:
                    audio_arr = audio_arr.mean(axis=1)
                t_audio = torch.from_numpy(audio_arr).float()
            else:
                continue

            # 2. Resample to 16kHz if needed
            if sr != 16000 and len(t_audio) > 160:
                target_len = int(len(t_audio) * (16000 / sr))
                t_audio = F.interpolate(
                    t_audio.view(1, 1, -1), size=target_len, mode="linear", align_corners=False
                ).squeeze()

            # Filter outlier durations (0.8s to 15.0s)
            dur_sec = len(t_audio) / 16000.0
            if dur_sec < 0.8 or dur_sec > 15.0:
                continue

            # 3. Tokenize transcript
            clean_text = sanitize_transcript(s.get("text", ""))
            token_ids = tokenizer.encode(clean_text)
            if len(token_ids) == 0:
                continue

            # 4. Optional Spec/Audio Augmentation
            if augment:
                t_audio = apply_audio_augmentation(t_audio)

            dialect_idx = s.get("dialect_idx", 3)  # default to Standard Marathi (3)
            valid.append((t_audio, torch.tensor(token_ids, dtype=torch.long), dialect_idx, clean_text))
        except Exception:
            continue

    if not valid:
        return None

    audio_tensors, token_tensors, dialect_indices, texts = zip(*valid)

    # Pad audio to max length in batch
    audio_lengths = torch.tensor([len(a) for a in audio_tensors], dtype=torch.long)
    max_audio_len = audio_lengths.max().item()
    padded_audio = torch.zeros(len(valid), max_audio_len, dtype=torch.float32)
    for i, a in enumerate(audio_tensors):
        padded_audio[i, :len(a)] = a

    target_lengths = torch.tensor([len(t) for t in token_tensors], dtype=torch.long)
    flat_targets = torch.cat(token_tensors)
    dialect_tensor = torch.tensor(dialect_indices, dtype=torch.long)

    return {
        "audio": padded_audio,
        "audio_lengths": audio_lengths,
        "audio_lens": audio_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "target_lens": target_lengths,
        "dialect_indices": dialect_tensor,
        "texts": list(texts),
    }


class MultiSourceCalibrationLoader:
    """
    Threaded prefetch loader sampling across RESPIN 5%, OpenSLR64, Kathbath,
    and the Pretraining Corpus.
    """

    def __init__(
        self,
        tokenizer: MarathiTokenizer,
        batch_size: int = 24,
        weights: List[float] = [0.35, 0.15, 0.35, 0.15],
        queue_size: int = 4,
        augment: bool = True,
        pretrain_stream_enabled: bool = True,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.weights = weights
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.augment = augment

        # 1. Source 0: RESPIN 5% Held-Out Split
        print("[MultiSourceLoader] Loading Source 1/4: RESPIN 5% Held-out Split...")
        _, calib_utts = build_or_load_speaker_split()
        self.respin_pool = calib_utts
        print(f"  -> RESPIN 5%: {len(self.respin_pool):,} utterances")

        # 2. Source 1: OpenSLR64
        print("[MultiSourceLoader] Loading Source 2/4: OpenSLR64 Marathi...")
        slr_manifest = "data/openslr64_marathi/manifest.json"
        slr_dir = "data/openslr64_marathi"
        if os.path.exists(slr_manifest):
            self.openslr_pool = load_kathbath_manifest(slr_manifest, slr_dir)
            for item in self.openslr_pool:
                item["dialect_idx"] = 3
        else:
            self.openslr_pool = []
        print(f"  -> OpenSLR64: {len(self.openslr_pool):,} utterances")

        # 3. Source 2: Kathbath Marathi
        print("[MultiSourceLoader] Loading Source 3/4: AI4Bharat Kathbath...")
        kb_manifest = "data/vistaar_benchmarks/kathbath_train/marathi/manifest.json"
        kb_dir = "data/vistaar_benchmarks/kathbath_train/marathi"
        if os.path.exists(kb_manifest):
            self.kathbath_pool = load_kathbath_manifest(kb_manifest, kb_dir)
            for item in self.kathbath_pool:
                item["dialect_idx"] = 3
        else:
            self.kathbath_pool = []
        print(f"  -> Kathbath: {len(self.kathbath_pool):,} utterances")

        # 4. Source 3: Pretraining Streaming Corpus
        self.pretrain_iter: Optional[Iterator[Dict[str, Any]]] = None
        if pretrain_stream_enabled:
            print("[MultiSourceLoader] Initializing Source 4/4: Pretraining Corpus Stream (Shrutilipi & Vaani)...")
            try:
                from pretraining.dataset_loader import get_combined_stream
                self.pretrain_iter = iter(get_combined_stream(
                    include_shrutilipi=True,
                    include_vaani=True,
                    vaani_dialects=["Marathi", "Konkani", "Malvani", "Khandeshi", "Powari", "Lambani"],
                    shrutilipi_dialects=["marathi", "konkani"],
                ))
                print("  -> Pretraining Stream: Online")
            except Exception as e:
                print(f"  -> Pretraining Stream Unavailable ({e}), falling back to local corpora.")
                self.pretrain_iter = None

        self.sources = [
            self.respin_pool,
            self.openslr_pool,
            self.kathbath_pool,
            self.pretrain_iter,
        ]

        self.worker = Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _sample_item(self, source_id: int) -> Optional[Dict[str, Any]]:
        source = self.sources[source_id]
        if source_id == 3:
            # Stream from pretraining corpus
            if self.pretrain_iter is not None:
                try:
                    for _ in range(5):
                        item = next(self.pretrain_iter)
                        txt = sanitize_transcript(item.get("text", ""))
                        if txt and len(txt) > 2:
                            return {
                                "audio": item.get("audio"),
                                "sampling_rate": item.get("sampling_rate", 16000),
                                "text": txt,
                                "dialect_idx": 3,
                            }
                except Exception:
                    pass
            # Fallback to local pool if stream yields empty or errors
            source_id = random.choice([0, 2])
            source = self.sources[source_id]

        if source and len(source) > 0:
            return random.choice(source)
        return None

    def _worker_loop(self):
        source_indices = [0, 1, 2, 3]
        while not self.stopped:
            batch_samples = []
            while len(batch_samples) < self.batch_size and not self.stopped:
                src_id = random.choices(source_indices, weights=self.weights)[0]
                item = self._sample_item(src_id)
                if item is not None:
                    batch_samples.append(item)

            if self.stopped:
                break

            batch = collate_multi_source_batch(batch_samples, self.tokenizer, augment=self.augment)
            if batch is not None:
                self.queue.put(batch)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        return self.queue.get()

    def close(self):
        self.stopped = True
