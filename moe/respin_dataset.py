"""
Stage 4: RESPIN Multi-Dialect Dataset Loader with Speaker-Disjoint Partitioning.
Loads from local IISc_RESPIN_train_mr_clean:
  - 95% Speakers: Stage 4 MoE Dialect Adaptation
  - 5% Speakers: Held out for Stage 5 Calibration
  - D1 (Malvani / Konkan) -> Dialect ID 0
  - D2 (Ahirani / Khandesh) -> Dialect ID 1
  - D4 (Varhadi / Vidarbha) -> Dialect ID 2
  - D3 (Standard Marathi) -> Dialect ID 3 (Uses Combining Shared Trunk)
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from queue import Queue
from threading import Thread

import soundfile as sf
import torch
import torch.nn.functional as F

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript


DIALECT_MAP = {
    "D1": 0,  # Malvani / Konkan
    "D2": 1,  # Ahirani / Khandesh
    "D4": 2,  # Varhadi / Vidarbha
    "D3": 3,  # Standard Marathi (Combining Shared Trunk)
}


def build_or_load_speaker_split(
    respin_dir: str = r"D:\dialect-norm\IISc_RESPIN_train_mr_clean\IISc_RESPIN_train_mr_clean",
    cache_path: str = "moe/respin_split_cache.json",
    train_speaker_ratio: float = 0.95,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Creates a strict speaker-disjoint partition of RESPIN Marathi data.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        print(f"[Dataset] Loading cached speaker split from: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["train"], data["calibration"]

    meta_json_path = Path(respin_dir) / "meta_train_mr_clean.json"
    print(f"[Dataset] Parsing metadata from: {meta_json_path}")
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Group utterances by speaker
    speaker_to_utts = {}
    for uid, info in meta.items():
        spk = info.get("speaker_id")
        if not spk:
            continue
        if spk not in speaker_to_utts:
            speaker_to_utts[spk] = []

        dialect = info.get("dialect", "D3")
        wav_rel = info.get("wav_path", "")
        wav_abs = str(Path(respin_dir) / wav_rel)
        text = sanitize_transcript(info.get("text", ""))

        if not text:
            continue

        speaker_to_utts[spk].append({
            "uid": uid,
            "speaker_id": spk,
            "dialect": dialect,
            "dialect_idx": DIALECT_MAP.get(dialect, 3),
            "wav_path": wav_abs,
            "text": text,
            "duration": info.get("duration", 0.0),
        })

    all_speakers = sorted(list(speaker_to_utts.keys()))
    random.seed(seed)
    random.shuffle(all_speakers)

    num_train_spk = int(len(all_speakers) * train_speaker_ratio)
    train_speakers = set(all_speakers[:num_train_spk])
    calib_speakers = set(all_speakers[num_train_spk:])

    train_utts = []
    calib_utts = []

    for spk, utts in speaker_to_utts.items():
        if spk in train_speakers:
            train_utts.extend(utts)
        else:
            calib_utts.extend(utts)

    print(f"[Dataset Split] Total Speakers: {len(all_speakers)}")
    print(f"  Train: {len(train_speakers)} speakers | {len(train_utts)} utterances")
    print(f"  Calibration (Held-out): {len(calib_speakers)} speakers | {len(calib_utts)} utterances")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"train": train_utts, "calibration": calib_utts}, f)
    print(f"[Dataset] Saved split cache to: {cache_file}")

    return train_utts, calib_utts


def apply_moe_audio_augmentation(
    waveform: torch.Tensor,
    enable_speed: bool = True,
    enable_noise: bool = True,
) -> torch.Tensor:
    """Applies speed perturbation (0.9x, 1.0x, 1.1x) and mild additive noise."""
    if enable_speed and random.random() < 0.5:
        speed = random.choice([0.9, 1.1])
        new_len = max(160, int(len(waveform) / speed))
        waveform = F.interpolate(waveform.view(1, 1, -1), size=new_len, mode="linear", align_corners=False).squeeze()

    if enable_noise and random.random() < 0.3:
        snr_db = random.uniform(15.0, 30.0)
        noise = torch.randn_like(waveform)
        sig_p = waveform.norm(p=2)
        noise_p = noise.norm(p=2)
        scale = (sig_p / (noise_p + 1e-8)) * (10.0 ** (-snr_db / 20.0))
        waveform = waveform + scale * noise

    return waveform


def collate_moe_batch(
    samples: List[Dict[str, Any]],
    tokenizer: MarathiTokenizer,
    augment: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collates parallel audio wavs, Devanagari text tokens, and dialect IDs.
    """
    valid = []
    for s in samples:
        try:
            audio, sr = sf.read(s["wav_path"], dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Must be 16kHz
            if sr != 16000:
                continue
            token_ids = tokenizer.encode(s["text"])
            if len(token_ids) == 0:
                continue
            t_audio = torch.from_numpy(audio)
            if augment:
                t_audio = apply_moe_audio_augmentation(t_audio)
            valid.append((t_audio, torch.tensor(token_ids, dtype=torch.long), s["dialect_idx"], s["text"]))
        except Exception:
            continue

    if not valid:
        return None

    audio_tensors, token_tensors, dialect_indices, texts = zip(*valid)

    # Pad audio to max length
    audio_lengths = torch.tensor([len(a) for a in audio_tensors], dtype=torch.long)
    max_audio_len = audio_lengths.max().item()
    padded_audio = torch.zeros(len(valid), max_audio_len, dtype=torch.float32)
    for i, a in enumerate(audio_tensors):
        padded_audio[i, :len(a)] = a

    # Concatenate targets for PyTorch CTCLoss
    target_lengths = torch.tensor([len(t) for t in token_tensors], dtype=torch.long)
    flat_targets = torch.cat(token_tensors)
    dialect_tensor = torch.tensor(dialect_indices, dtype=torch.long)

    return {
        "audio": padded_audio,
        "audio_lengths": audio_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "dialect_indices": dialect_tensor,
        "texts": list(texts),
    }


class LocalRESPINPrefetchLoader:
    """
    Threaded fast non-blocking prefetch loader with length-bucketing optimization.
    Groups utterances into duration-sorted buckets to eliminate GPU zero-padding waste.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: MarathiTokenizer,
        batch_size: int = 32,
        queue_size: int = 4,
        shuffle: bool = True,
        bucket_multiplier: int = 4,
    ):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_multiplier = bucket_multiplier
        self.queue = Queue(maxsize=queue_size)
        self._stop_signal = False

        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        pool = list(self.samples)
        bucket_size = self.batch_size * self.bucket_multiplier

        while not self._stop_signal:
            if self.shuffle:
                random.shuffle(pool)

            # Process pool in chunks of bucket_size, sort by duration, and batch
            for i in range(0, len(pool), bucket_size):
                if self._stop_signal:
                    break
                chunk = pool[i : i + bucket_size]
                # Length-bucketing sort to minimize intra-batch padding
                chunk.sort(key=lambda x: x.get("duration", 0.0))

                for j in range(0, len(chunk), self.batch_size):
                    if self._stop_signal:
                        break
                    batch_samples = chunk[j : j + self.batch_size]
                    batch = collate_moe_batch(batch_samples, self.tokenizer)
                    if batch is not None:
                        self.queue.put(batch)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        return self.queue.get()

    def close(self):
        self._stop_signal = True
