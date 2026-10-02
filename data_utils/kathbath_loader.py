import json
import os
from pathlib import Path
from queue import Queue
from threading import Thread
import random
import torch
import soundfile as sf
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from data_utils.tokenizer import MarathiTokenizer

def apply_audio_augmentation(
    waveform: torch.Tensor,
    enable_speed: bool = True,
    enable_noise: bool = True,
) -> torch.Tensor:
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

def collate_kathbath_batch(
    samples: List[Dict[str, Any]],
    tokenizer: MarathiTokenizer,
    augment: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    valid = []
    for s in samples:
        try:
            audio, sr = sf.read(s["wav_path"], dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            t_audio = torch.from_numpy(audio)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                # 48kHz -> 16kHz is exactly 1/3
                target_len = int(len(t_audio) * (16000 / sr))
                t_audio = F.interpolate(t_audio.view(1, 1, -1), size=target_len, mode="linear", align_corners=False).squeeze()
                
            token_ids = tokenizer.encode(s["text"])
            if len(token_ids) == 0:
                continue
                
            if augment:
                t_audio = apply_audio_augmentation(t_audio)
            valid.append((t_audio, torch.tensor(token_ids, dtype=torch.long), s.get("dialect_idx", 3), s["text"]))
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
        "audio_lens": audio_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "target_lens": target_lengths,
        "dialect_indices": dialect_tensor,
        "texts": list(texts),
    }

def load_kathbath_manifest(manifest_path: str, data_dir: str) -> List[Dict[str, Any]]:
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rel_path = item["audio_filepath"]
            # Fix relative path
            if rel_path.startswith("kathbath_train/marathi/"):
                rel_path = rel_path[len("kathbath_train/marathi/"):]
            elif rel_path.startswith("kathbath_noisy/marathi/"):
                rel_path = rel_path[len("kathbath_noisy/marathi/"):]
            wav_path = os.path.join(data_dir, rel_path)
            samples.append({
                "wav_path": wav_path,
                "text": item["text"],
                "duration": item.get("duration", 0.0),
                "dialect_idx": 3 # Default to standard Marathi for Kathbath
            })
    return samples

class LocalKathbathPrefetchLoader:
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

            for i in range(0, len(pool), bucket_size):
                if self._stop_signal:
                    break
                chunk = pool[i : i + bucket_size]
                chunk.sort(key=lambda x: x.get("duration", 0.0))

                for j in range(0, len(chunk), self.batch_size):
                    if self._stop_signal:
                        break
                    batch_samples = chunk[j : j + self.batch_size]
                    batch = collate_kathbath_batch(batch_samples, self.tokenizer)
                    if batch is not None:
                        self.queue.put(batch)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        return self.queue.get()

    def close(self):
        self._stop_signal = True
