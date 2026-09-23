"""
Marathi Devanagari Character/Grapheme Tokenizer for CTC ASR.
Supports all Marathi vowels, consonants, matras, halant/virama, conjuncts, numbers, and CTC blanks.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# Annotation metadata cleaner for Vaani & Shrutilipi transcriber tags
ANNOTATION_TAG_PATTERN = re.compile(
    r'<[^>]+>|'          # XML tags: <noise>, </noise>, <pause>, <insect_noise>, etc.
    r'\[[^\]]+\]|'       # Bracket tags: [breathing], [laughter], etc.
    r'\{[^}]+\}',        # Curly brace tags: {coffee}, {sweater}, etc.
    re.IGNORECASE
)


def sanitize_transcript(text: str) -> str:
    """
    Strip all transcriber annotation metadata from Vaani/Shrutilipi text.
    Removes tags like <noise>, </noise>, <pause>, [breathing], {coffee},
    and normalizes excessive whitespace.
    """
    if not text:
        return ""
    text = ANNOTATION_TAG_PATTERN.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Standard Devanagari Unicode Components for Marathi
DEVANAGARI_VOWELS = [
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ", "ॲ", "ऑ"
]

DEVANAGARI_CONSONANTS = [
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व", "श", "ष", "स", "ह", "ळ",
    "क्ष", "ज्ञ"  # Common Marathi conjuncts
]

DEVANAGARI_MATRAS = [
    "ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॢ", "ॣ", "े", "ै", "ो", "ौ", "ॅ", "ॉ"
]

DEVANAGARI_SIGNS = [
    "ं",  # Anusvara
    "ः",  # Visarga
    "ँ",  # Chandrabindu
    "़",  # Nukta
    "्",  # Virama / Halant
    "ऽ",  # Avagraha
]

DIGITS = [
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

PUNCTUATION = [" ", "।", ".", ",", "?", "!", "-", "'", '"']

SPECIAL_TOKENS = ["<blank>", "<unk>", "<pad>"]


class MarathiTokenizer:
    """Character-level CTC tokenizer for Marathi ASR."""

    def __init__(self, vocab_path: Optional[str] = None):
        if vocab_path and Path(vocab_path).exists():
            self.load_vocab(vocab_path)
        else:
            self._build_default_vocab()
            if vocab_path:
                self.save_vocab(vocab_path)

    def _build_default_vocab(self):
        """Constructs the canonical Marathi character vocabulary."""
        tokens = list(SPECIAL_TOKENS)
        # Add whitespace explicitly
        tokens.append(" ")

        # Combine all characters preserving linguistic order
        all_chars = (
            DEVANAGARI_VOWELS
            + DEVANAGARI_CONSONANTS
            + DEVANAGARI_MATRAS
            + DEVANAGARI_SIGNS
            + DIGITS
            + [p for p in PUNCTUATION if p != " "]
        )

        for ch in all_chars:
            if ch not in tokens:
                tokens.append(ch)

        self.token2id: Dict[str, int] = {token: idx for idx, token in enumerate(tokens)}
        self.id2token: Dict[int, str] = {idx: token for idx, token in enumerate(tokens)}
        self.blank_id = self.token2id["<blank>"]
        self.unk_id = self.token2id["<unk>"]
        self.pad_id = self.token2id["<pad>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def encode(self, text: str, sanitize: bool = True) -> List[int]:
        """Encodes text into a list of token IDs, optionally sanitizing transcriber annotation tags."""
        if sanitize:
            text = sanitize_transcript(text)
        ids = []
        for char in text:
            ids.append(self.token2id.get(char, self.unk_id))
        return ids

    def decode(self, token_ids: List[int], remove_special: bool = True) -> str:
        """Decodes token IDs into a raw string."""
        chars = []
        for tid in token_ids:
            token = self.id2token.get(tid, "")
            if remove_special and token in SPECIAL_TOKENS:
                continue
            chars.append(token)
        return "".join(chars)

    def ctc_decode(self, token_ids: List[int]) -> str:
        """Applies CTC greedy collapsing: removes consecutive duplicate tokens and blanks."""
        collapsed = []
        prev = None
        for tid in token_ids:
            if tid != prev:
                if tid != self.blank_id:
                    collapsed.append(tid)
                prev = tid
        return self.decode(collapsed, remove_special=True)

    def save_vocab(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "token2id": self.token2id,
                "vocab_size": self.vocab_size,
                "blank_id": self.blank_id,
                "unk_id": self.unk_id,
                "pad_id": self.pad_id,
            }, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.token2id = data["token2id"]
            self.id2token = {int(v): k for k, v in self.token2id.items()}
            self.blank_id = data["blank_id"]
            self.unk_id = data["unk_id"]
            self.pad_id = data["pad_id"]


if __name__ == "__main__":
    tok = MarathiTokenizer("data_utils/vocab.json")
    print(f"Constructed Marathi Vocab Size: {tok.vocab_size}")
    sample_text = "नमस्कार! मी महाराष्ट्राचा रहिवासी आहे."
    encoded = tok.encode(sample_text)
    decoded = tok.decode(encoded)
    print(f"Sample Text: {sample_text}")
    print(f"Encoded IDs ({len(encoded)}): {encoded[:10]}...")
    print(f"Decoded: {decoded}")
    assert sample_text == decoded, "Tokenizer encode/decode mismatch!"
    print("[Success] Marathi Tokenizer verified!")
