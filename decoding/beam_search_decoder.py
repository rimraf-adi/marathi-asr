"""
Marathi CTC Beam Search Decoder using pyctcdecode.
Supports prefix-tree unigram dictionary beam search and optional KenLM language model integration.
Gracefully falls back to greedy CTC decode if pyctcdecode encounters any issues.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union, Collection
import numpy as np
import torch

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript

try:
    import pyctcdecode
    PYCTCDECODE_AVAILABLE = True
except ImportError:
    PYCTCDECODE_AVAILABLE = False


class MarathiBeamSearchDecoder:
    """
    Beam search decoder with lexicon prefix-tree and optional KenLM for Marathi CTC ASR.
    """

    def __init__(
        self,
        tokenizer: MarathiTokenizer,
        unigram_words: Optional[Collection[str]] = None,
        kenlm_path: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 1.5,
    ):
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.kenlm_path = kenlm_path
        self.decoder = None

        if PYCTCDECODE_AVAILABLE:
            self._init_pyctcdecode(unigram_words)
        else:
            print("[BeamSearchDecoder] pyctcdecode not installed; defaulting to greedy CTC decoding.")

    def _init_pyctcdecode(self, unigram_words: Optional[Collection[str]] = None):
        """Builds pyctcdecode.BeamSearchDecoderCTC with custom Marathi vocabulary labels."""
        try:
            # Construct labels matching MarathiTokenizer token IDs:
            # Blank ID (0) must be represented as "" for pyctcdecode.
            labels = [self.tokenizer.id2token[i] for i in range(self.tokenizer.vocab_size)]
            labels[self.tokenizer.blank_id] = ""

            unigrams_list = None
            if unigram_words:
                unigrams_list = [w.strip() for w in unigram_words if len(w.strip()) > 0]

            self.decoder = pyctcdecode.build_ctcdecoder(
                labels=labels,
                kenlm_model_path=self.kenlm_path if (self.kenlm_path and os.path.exists(self.kenlm_path)) else None,
                unigrams=unigrams_list,
                alpha=self.alpha,
                beta=self.beta,
            )
            print(f"[BeamSearchDecoder] Initialized pyctcdecode BeamSearchDecoderCTC (vocab size: {len(labels)}, unigrams: {len(unigrams_list) if unigrams_list else 0})")
        except Exception as e:
            print(f"[BeamSearchDecoder] Warning: pyctcdecode initialization failed: {e}. Falling back to greedy decoding.")
            self.decoder = None

    def decode(self, logits: Union[np.ndarray, torch.Tensor], beam_width: int = 32) -> str:
        """
        Decodes a single 2D logit or log-probability matrix (T, vocab_size).
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().float().numpy()

        if self.decoder is not None:
            try:
                decoded = self.decoder.decode(logits, beam_width=beam_width)
                return sanitize_transcript(decoded)
            except Exception:
                pass

        # Fallback to greedy CTC decode
        token_ids = np.argmax(logits, axis=-1).tolist()
        return sanitize_transcript(self.tokenizer.ctc_decode(token_ids))

    def decode_batch(
        self,
        logits_batch: Union[np.ndarray, torch.Tensor],
        lengths: Optional[List[int]] = None,
        beam_width: int = 32,
    ) -> List[str]:
        """
        Decodes a 3D batch of logits (B, T, vocab_size).
        """
        if isinstance(logits_batch, torch.Tensor):
            logits_batch = logits_batch.detach().cpu().float().numpy()

        results = []
        b_size = logits_batch.shape[0]
        for b in range(b_size):
            cur_logits = logits_batch[b]
            if lengths is not None and b < len(lengths):
                cur_logits = cur_logits[:lengths[b]]
            results.append(self.decode(cur_logits, beam_width=beam_width))
        return results


def load_marathi_wordlist(cache_path: str = "moe/respin_split_cache.json", max_words: int = 50000) -> List[str]:
    """
    Extracts high-frequency unique Marathi words from local dataset caches to seed the beam search lexicon.
    """
    import json
    import re

    cache_file = Path(cache_path)
    if not cache_file.exists():
        return []

    words_freq = {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for split in ["train", "calibration"]:
            for item in data.get(split, []):
                text = sanitize_transcript(item.get("text", ""))
                tokens = re.findall(r'[\u0900-\u097F]+', text)
                for w in tokens:
                    if len(w) > 1:
                        words_freq[w] = words_freq.get(w, 0) + 1

        sorted_words = sorted(words_freq.keys(), key=lambda w: words_freq[w], reverse=True)
        return sorted_words[:max_words]
    except Exception as e:
        print(f"[Wordlist] Could not load wordlist from cache: {e}")
        return []
