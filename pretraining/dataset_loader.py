"""
Unified dataset streaming pipeline for Marathi & all related dialects
from both ai4bharat/Shrutilipi and ARTPARK-IISc/Vaani.
Features:
  - Local small-chunk caching with automatic network relay upon cache exhaustion
  - HF rate-limit (HTTP 429) resilience with exponential backoff
  - Clean epoch-based iteration (no infinite streaming)
  - Windows & Mac cross-platform cache directory resolution
"""

import os
import sys
import time
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Callable
import torch
import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Set Hugging Face cache path (cross-platform: D:\ drive on Windows if present, else standard cache)
default_hf_home = r"D:\huggingface_cache" if sys.platform == "win32" and os.path.exists("D:\\") else os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", default_hf_home))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from datasets import load_dataset
from data_utils.tokenizer import sanitize_transcript

# Increase HF Hub timeout for streaming
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

# Config mapping for dialects
DIALECT_REGISTRY = {
    "shrutilipi": {
        "marathi": {"split": "train", "dialect_tag": "mr_standard"},
        "konkani": {"split": "train", "dialect_tag": "konkani"},
    },
    "vaani": {
        "Marathi": {"split": "train", "dialect_tag": "mr_general"},
        "Konkani": {"split": "train", "dialect_tag": "konkani"},
        "Malvani": {"split": "train", "dialect_tag": "mr_malvani"},
        "Khandeshi": {"split": "train", "dialect_tag": "mr_khandeshi"},
        "Powari": {"split": "train", "dialect_tag": "mr_powari"},
        "Lambani": {"split": "train", "dialect_tag": "mr_lambani"},
    },
}


def stream_shrutilipi(configs: List[str] = None) -> Iterator[Dict[str, Any]]:
    """Stream audio examples from Shrutilipi."""
    if configs is None:
        configs = list(DIALECT_REGISTRY["shrutilipi"].keys())

    for config in configs:
        meta = DIALECT_REGISTRY["shrutilipi"][config]
        ds = load_dataset(
            "ai4bharat/Shrutilipi",
            config,
            split=meta["split"],
            streaming=True,
            token=HF_TOKEN,
        )
        for item in ds:
            audio_data = item["audio_filepath"]
            yield {
                "source": "shrutilipi",
                "config": config,
                "dialect": meta["dialect_tag"],
                "audio": audio_data["array"],
                "sampling_rate": audio_data["sampling_rate"],
                "duration": item.get("duration", len(audio_data["array"]) / audio_data["sampling_rate"]),
                "text": sanitize_transcript(item.get("text", "")),
                "speaker_id": None,
                "district": None,
            }


def stream_vaani(configs: List[str] = None) -> Iterator[Dict[str, Any]]:
    """Stream audio examples from ARTPARK-IISc/Vaani."""
    if configs is None:
        configs = list(DIALECT_REGISTRY["vaani"].keys())

    for config in configs:
        meta = DIALECT_REGISTRY["vaani"][config]
        ds = load_dataset(
            "ARTPARK-IISc/Vaani",
            config,
            split=meta["split"],
            streaming=True,
            token=HF_TOKEN,
        )
        for item in ds:
            audio_data = item["audio"]
            sr = audio_data.get("sampling_rate", 16000)
            arr = audio_data["array"]
            dur = item.get("duration") or (len(arr) / sr if sr else 0.0)

            raw_txt = item.get("transcript", "") if item.get("isTranscriptionAvailable") == "Yes" else ""
            yield {
                "source": "vaani",
                "config": config,
                "dialect": meta["dialect_tag"],
                "audio": arr,
                "sampling_rate": sr,
                "duration": float(dur),
                "text": sanitize_transcript(raw_txt),
                "speaker_id": item.get("speakerID"),
                "district": item.get("district"),
            }


def get_combined_stream(
    include_shrutilipi: bool = True,
    include_vaani: bool = True,
    vaani_dialects: List[str] = None,
    shrutilipi_dialects: List[str] = None,
    interleave: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield unified samples interleaved across sources to prevent distribution shift."""
    iterators = []
    if include_shrutilipi:
        iterators.append(iter(stream_shrutilipi(shrutilipi_dialects)))
    if include_vaani:
        iterators.append(iter(stream_vaani(vaani_dialects)))

    if not iterators:
        return

    if not interleave or len(iterators) == 1:
        for it in iterators:
            yield from it
        return

    # Round-robin interleaving across sources
    while iterators:
        active_iterators = []
        for it in iterators:
            try:
                yield next(it)
                active_iterators.append(it)
            except StopIteration:
                pass
        iterators = active_iterators


class CachedRelayedStream:
    """
    Two-Tier Data Feeder with Local Chunk Caching & Dynamic Network Relay:
    1. Reads samples directly from small local disk cache shards (chunk_00000.pt, etc.).
       - Fast throughput, zero network latency, zero HF API calls.
    2. Once the local cache buffer is exhausted, relays to the network stream:
       - Fetches small chunks (default: 500 samples) from Hugging Face with exponential backoff on rate limits (HTTP 429).
       - Persists chunk to disk as a serialized torch tensor shard.
       - Cleanly pauses network connection and resumes serving from the local cache.
    3. Guarantees deterministic, finite training for X epochs:
       - No infinite loops.
       - Epoch boundaries cleanly signal StopIteration when the dataset pass completes.
       - Re-uses cached chunks on disk across epochs without re-downloading from network.
    """

    def __init__(
        self,
        stream_factory: Optional[Callable[[], Iterator[Dict[str, Any]]]] = None,
        cache_dir: str = "data_cache/pretrain",
        chunk_size: int = 10000,
        max_cached_chunks: Optional[int] = 10,
        min_duration: float = 1.5,
        max_duration: float = 14.0,
    ):
        self.stream_factory = stream_factory
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self.max_cached_chunks = max_cached_chunks
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._network_stream: Optional[Iterator[Dict[str, Any]]] = None
        self._network_exhausted: bool = False
        self._manifest_file = self.cache_dir / "cache_manifest.json"
        self._total_chunks: Optional[int] = self._load_manifest()

    def _load_manifest(self) -> Optional[int]:
        if self._manifest_file.exists():
            try:
                import json
                with open(self._manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("total_chunks")
            except Exception:
                return None
        return None

    def _save_manifest(self, total_chunks: int):
        self._total_chunks = total_chunks
        try:
            import json
            with open(self._manifest_file, "w", encoding="utf-8") as f:
                json.dump({"total_chunks": total_chunks}, f, indent=2)
        except Exception:
            pass

    def _get_chunk_path(self, chunk_idx: int) -> Path:
        return self.cache_dir / f"chunk_{chunk_idx:05d}.pt"

    def _fetch_next_network_chunk(self, chunk_idx: int) -> Optional[List[Dict[str, Any]]]:
        """Relays to network stream to fetch chunk_size samples with rate-limit backoff."""
        if self._network_exhausted or self.stream_factory is None:
            return None

        if self._network_stream is None:
            print(f"[Network Relay] Opening remote dataset stream for shard {chunk_idx:05d}...", flush=True)
            self._network_stream = iter(self.stream_factory())

        chunk_items: List[Dict[str, Any]] = []
        backoff_sec = 2.0
        consecutive_failures = 0
        max_retries = 6

        print(f"[Network Relay] Local cache exhausted. Relaying to network for next {self.chunk_size} samples...", flush=True)

        while len(chunk_items) < self.chunk_size:
            try:
                item = next(self._network_stream)
                dur = float(item.get("duration", 0.0))
                if dur < self.min_duration or dur > self.max_duration:
                    continue

                audio_arr = item["audio"]
                if hasattr(audio_arr, "numpy"):
                    audio_arr = audio_arr.numpy()
                if not isinstance(audio_arr, np.ndarray):
                    audio_arr = np.array(audio_arr, dtype=np.float32)
                elif audio_arr.dtype != np.float32:
                    audio_arr = audio_arr.astype(np.float32)

                sample_dict = {
                    "source": item.get("source", "unknown"),
                    "config": item.get("config", "unknown"),
                    "dialect": item.get("dialect", "mr_standard"),
                    "audio": audio_arr,
                    "sampling_rate": int(item.get("sampling_rate", 16000)),
                    "duration": dur,
                    "text": item.get("text", ""),
                    "speaker_id": item.get("speaker_id"),
                    "district": item.get("district"),
                }
                chunk_items.append(sample_dict)
                backoff_sec = 2.0
                consecutive_failures = 0

            except StopIteration:
                self._network_exhausted = True
                print(f"[Network Relay] Reached end of remote stream.", flush=True)
                break
            except Exception as e:
                consecutive_failures += 1
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many requests" in err_str
                prefix = "[HF Rate Limit 429]" if is_rate_limit else "[Network Error]"
                print(f"{prefix} {e}. Backing off for {backoff_sec:.1f}s (retry {consecutive_failures}/{max_retries})...", file=sys.stderr, flush=True)
                if consecutive_failures >= max_retries:
                    print(f"[Network Relay Error] Exceeded max retries ({max_retries}). Halting chunk fetch.", file=sys.stderr, flush=True)
                    break
                time.sleep(backoff_sec)
                backoff_sec = min(backoff_sec * 2.0, 60.0)

        if not chunk_items:
            if self._network_exhausted:
                self._save_manifest(chunk_idx)
            return None

        chunk_path = self._get_chunk_path(chunk_idx)
        torch.save(chunk_items, chunk_path)
        sz_mb = chunk_path.stat().st_size / (1024**2)
        print(f"[Network Relay] Cached {len(chunk_items)} samples to {chunk_path.name} ({sz_mb:.1f} MB)", flush=True)

        if self._network_exhausted:
            self._save_manifest(chunk_idx + 1)

        self._enforce_cache_quota()
        return chunk_items

    def _enforce_cache_quota(self):
        """Maintains cache size within max_cached_chunks to avoid disk saturation."""
        if self.max_cached_chunks is None or self.max_cached_chunks <= 0:
            return
        all_chunks = sorted(self.cache_dir.glob("chunk_*.pt"))
        if len(all_chunks) > self.max_cached_chunks:
            excess = len(all_chunks) - self.max_cached_chunks
            for p in all_chunks[:excess]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    def iter_epoch(self, epoch: int) -> Iterator[Dict[str, Any]]:
        """
        Iterates through the dataset for a single epoch.
        Serves from cache when available, relays to network when cache is exhausted.
        Terminates cleanly at the end of the epoch.
        """
        chunk_idx = 0
        total_samples = 0
        self._network_stream = None
        self._network_exhausted = False

        while True:
            if self._total_chunks is not None and chunk_idx >= self._total_chunks:
                break

            chunk_path = self._get_chunk_path(chunk_idx)

            if chunk_path.exists():
                try:
                    items = torch.load(chunk_path, weights_only=False)
                except Exception as e:
                    print(f"[Cache Warning] Failed reading {chunk_path.name}: {e}. Relaying to network...", file=sys.stderr, flush=True)
                    chunk_path.unlink(missing_ok=True)
                    items = self._fetch_next_network_chunk(chunk_idx)
            else:
                items = self._fetch_next_network_chunk(chunk_idx)

            if items is None or len(items) == 0:
                break

            for item in items:
                yield item
                total_samples += 1

            chunk_idx += 1

        print(f"[Dataset] Epoch {epoch} stream completed. Processed {total_samples:,} samples.", flush=True)


if __name__ == "__main__":
    print("Testing unified streaming across Shrutilipi & Vaani dialects...\n")

    # Sample 1 item from each dialect
    for source_name, dialect_dict in DIALECT_REGISTRY.items():
        print(f"\n{'='*55}\nSource: {source_name.upper()}\n{'='*55}")
        for cfg in dialect_dict.keys():
            if source_name == "shrutilipi":
                sample_iter = stream_shrutilipi([cfg])
            else:
                sample_iter = stream_vaani([cfg])

            sample = next(iter(sample_iter))
            print(f"[{cfg:10s}] dialect: {sample['dialect']:14s} | "
                  f"audio shape: {str(sample['audio'].shape):12s} | "
                  f"dur: {sample['duration']:5.2f}s | "
                  f"district: {str(sample['district']):14s} | "
                  f"text: {sample['text'][:50] if sample['text'] else '[Unlabeled]'}")
