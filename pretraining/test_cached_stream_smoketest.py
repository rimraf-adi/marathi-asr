"""
Smoke Test Suite for CachedRelayedStream, ThreadedPrefetchLoader, and Epoch-based Pretraining.
Runs entirely in-memory with synthetic dummy audio.
DOES NOT download large datasets or execute heavy GPU workloads.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repo root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from pretraining.dataset_loader import CachedRelayedStream
from pretraining.prefetch_loader import ThreadedPrefetchLoader
from pretraining.train import collate_audio_batch, train_pretrain


def create_mock_sample(idx: int) -> dict:
    """Generates a synthetic 16kHz audio sample (2 seconds duration)."""
    return {
        "source": "mock_source",
        "config": "mock_cfg",
        "dialect": "mr_standard",
        "audio": np.random.randn(32000).astype(np.float32),
        "sampling_rate": 16000,
        "duration": 2.0,
        "text": f"नमुना ऑडिओ वाक्य {idx}",
        "speaker_id": f"spk_{idx % 3}",
        "district": "Pune",
    }


def mock_stream_generator(num_samples: int = 12):
    """Generator yielding synthetic audio samples."""
    for i in range(num_samples):
        yield create_mock_sample(i)


class TestCachedRelayedStreamSmoke(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_asr_cache_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chunking_and_network_relay(self):
        """Verifies that samples are chunked to disk, and served from cache on subsequent reads."""
        num_samples = 10
        chunk_size = 4
        call_count = {"calls": 0}

        def counting_generator():
            call_count["calls"] += 1
            for item in mock_stream_generator(num_samples):
                yield item

        stream = CachedRelayedStream(
            stream_factory=counting_generator,
            cache_dir=self.temp_dir,
            chunk_size=chunk_size,
            max_cached_chunks=10,
        )

        # Epoch 1: Should pull from counting_generator and write chunks to disk
        items_epoch1 = list(stream.iter_epoch(epoch=1))
        self.assertEqual(len(items_epoch1), num_samples)
        self.assertEqual(call_count["calls"], 1)

        # Verify disk shards exist: chunk_00000.pt, chunk_00001.pt, chunk_00002.pt
        shards = sorted(Path(self.temp_dir).glob("chunk_*.pt"))
        expected_shards = int(np.ceil(num_samples / chunk_size))
        self.assertEqual(len(shards), expected_shards)

        # Epoch 2: Counting generator should NOT be called at all because cache is populated
        def exploding_generator():
            raise AssertionError("Network stream called even though local cache exists!")

        stream.stream_factory = exploding_generator
        items_epoch2 = list(stream.iter_epoch(epoch=2))
        self.assertEqual(len(items_epoch2), num_samples)
        self.assertEqual(items_epoch1[0]["text"], items_epoch2[0]["text"])

    def test_prefetch_loader_clean_stop(self):
        """Verifies ThreadedPrefetchLoader terminates cleanly (no infinite hang) at stream end."""
        samples = [create_mock_sample(i) for i in range(6)]
        prefetcher = ThreadedPrefetchLoader(
            stream_iterator=iter(samples),
            batch_size=2,
            max_prefetch=2,
            collate_fn=collate_audio_batch,
            infinite=False,
        )

        batches = []
        for batch, raw_items in prefetcher:
            batches.append(batch)

        prefetcher.close()
        # 6 samples / batch_size 2 = 3 batches
        self.assertEqual(len(batches), 3)

    def test_pretraining_two_epochs_smoketest(self):
        """Runs a 2-epoch minimal pretraining smoketest with dummy model & data."""
        test_run_dir = os.path.join(self.temp_dir, "test_run")
        test_cache_dir = os.path.join(self.temp_dir, "cache")

        def dummy_factory():
            return mock_stream_generator(num_samples=4)

        # Should complete 2 epochs of 2 batches each (batch_size=2, num_samples=4) in ~1-2 seconds
        train_pretrain(
            run_dir=test_run_dir,
            epochs=2,
            total_steps=10,
            batch_size=2,
            lr=1e-3,
            warmup_steps=1,
            log_every=1,
            plot_every=999,
            eval_every=999,
            save_every=999,
            cache_dir=test_cache_dir,
            cache_chunk_size=2,
            max_cached_chunks=5,
            custom_stream_factory=dummy_factory,
        )

        # Verify checkpoints directory was created and contains saved models
        pts = list(Path(test_run_dir).glob("checkpoints/**/*.pt"))
        self.assertGreater(len(pts), 0)


if __name__ == "__main__":
    unittest.main()
