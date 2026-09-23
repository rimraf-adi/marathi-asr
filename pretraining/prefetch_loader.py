"""
High-Performance Deadlock-Free Streaming Prefetcher for Windows.

Solves the Windows PyTorch multiprocessing deadlock:
  1. Uses asynchronous background producer threads instead of spawned OS processes.
  2. Avoids Windows Named Pipe / IPC queue buffer deadlocks.
  3. Eliminates [WinError 10038] socket inheritance errors when streaming from Hugging Face.
  4. Keeps the RTX A5000 100% saturated by prefetching batches into a bounded queue.
  5. Provides clean epoch boundaries and StopIteration propagation without infinite loops.
"""

import os
import sys
import time
import queue
import threading
from typing import Iterator, Dict, Any, List, Optional
import torch

# Prevent OpenMP / MKL thread contention and Windows deadlocks
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class ThreadedPrefetchLoader:
    """
    An asynchronous background prefetcher that decouples network/disk I/O and batch collation
    from GPU training steps without using multiprocessing (which deadlocks on Windows).

    Args:
        stream_iterator: Single epoch iterator (preferred for finite epochs)
        stream_factory: Callable returning stream iterator
        batch_size: Number of audio samples per batch
        max_prefetch: Queue capacity (default: 8 batches buffered in memory)
        collate_fn: Function to pad and convert a list of audio items into a PyTorch batch
        device: Target torch device (e.g. 'cuda')
        infinite: If False (default), raises StopIteration when the stream/epoch ends.
    """

    def __init__(
        self,
        stream_iterator=None,
        stream_factory=None,
        batch_size: int = 16,
        max_prefetch: int = 8,
        collate_fn=None,
        device: torch.device = torch.device("cpu"),
        min_duration: float = 1.5,
        max_duration: float = 14.0,
        infinite: bool = False,
    ):
        self.stream_iterator = stream_iterator
        self.stream_factory = stream_factory
        self.batch_size = batch_size
        self.max_prefetch = max_prefetch
        self.collate_fn = collate_fn
        self.device = device
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.infinite = infinite

        self.queue = queue.Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()
        self.sentinel = object()  # Marker for end of stream

        # Start producer daemon thread
        self.worker_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.worker_thread.start()

    def _producer_loop(self):
        """Runs in background thread: reads stream, builds batches, pushes to queue."""
        batch_items = []
        consecutive_failures = 0
        max_retries = 10
        try:
            while not self.stop_event.is_set():
                try:
                    if self.stream_factory is not None:
                        stream = self.stream_factory()
                    else:
                        stream = self.stream_iterator
                except Exception as e:
                    consecutive_failures += 1
                    print(f"[Prefetcher] stream_factory failed ({consecutive_failures}/{max_retries}): {e}", file=sys.stderr)
                    if consecutive_failures >= max_retries:
                        break
                    time.sleep(min(5 * consecutive_failures, 30))
                    continue

                if stream is None:
                    break

                stream_exhausted = False
                for item in stream:
                    if self.stop_event.is_set():
                        break
                    try:
                        dur = item.get("duration", 0.0)
                    except Exception:
                        continue
                    if dur < self.min_duration or dur > self.max_duration:
                        continue

                    batch_items.append(item)
                    consecutive_failures = 0

                    if len(batch_items) >= self.batch_size:
                        if self.collate_fn is not None:
                            try:
                                batch = self.collate_fn(batch_items)
                            except Exception as e:
                                print(f"[Prefetcher] collate failed: {e}", file=sys.stderr)
                                batch_items = []
                                continue
                        else:
                            batch = batch_items

                        # Blocking put with timeout to allow clean shutdown
                        while not self.stop_event.is_set():
                            try:
                                self.queue.put((batch, list(batch_items)), timeout=0.5)
                                break
                            except queue.Full:
                                continue

                        batch_items = []
                else:
                    stream_exhausted = True

                # Flush final partial batch if any items remain
                if stream_exhausted and len(batch_items) > 0 and not self.stop_event.is_set():
                    try:
                        batch = self.collate_fn(batch_items) if self.collate_fn is not None else batch_items
                        self.queue.put((batch, list(batch_items)), timeout=1.0)
                    except Exception:
                        pass
                    batch_items = []

                # If finite (default), exit loop upon stream exhaustion
                if not self.infinite or self.stream_factory is None:
                    if stream_exhausted:
                        print(f"[Prefetcher] Stream pass completed.", file=sys.stderr)
                    break

                if consecutive_failures > 0:
                    time.sleep(2)

        except Exception as e:
            import traceback
            print(f"[Prefetcher Error] Worker thread encountered exception: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            self.queue.put((self.sentinel, None))

    def __iter__(self):
        return self

    def __next__(self):
        while not self.stop_event.is_set():
            try:
                item, raw_items = self.queue.get(timeout=2.0)
                if item is self.sentinel:
                    raise StopIteration
                return item, raw_items
            except queue.Empty:
                if not self.worker_thread.is_alive() and self.queue.empty():
                    raise StopIteration
                continue
        raise StopIteration

    def close(self):
        """Signal background thread to terminate and drain queue."""
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
