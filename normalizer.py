import re
import unicodedata

NORM_VERSION = "mr-norm-1"

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ANGLE_TAGS = re.compile(r"<[^>]*>")  # annotation tags like <noise>, <laughter>
_BRACES = re.compile(r"[{}]")  # keep the word, drop code-switch markers
_PUNCT = re.compile(r"[\u0964\u0965.,!?;:\"'()\[\]\-\u2013\u2014\u2026/\\_~`@#$%^&*+=|]")  # danda, double danda, ASCII punctuation
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d]")
_WS = re.compile(r"\s+")


def normalize(text: str, strip_zero_width: bool = True) -> str:
    """Standard frozen Marathi text normalizer (mr-norm-1).

    Rules:
    1. NFC unicode canonical decomposition followed by canonical composition.
    2. Remove annotation tags (<noise>, etc.).
    3. Remove code-switch curly braces while retaining word contents.
    4. Normalize Devanagari numerals (०-९) to ASCII digits (0-9) consistently in both refs and hyps.
    5. Strip dandas, double-dandas, and punctuation.
    6. Strip zero-width joiners and non-joiners (U+200B, U+200C, U+200D).
    7. Lowercase ASCII/Latin characters (handles code-mixed English words).
    8. Collapse multiple whitespace characters into a single space and strip edges.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = _ANGLE_TAGS.sub(" ", text)
    text = _BRACES.sub("", text)
    text = text.translate(_DEVANAGARI_DIGITS)
    text = _PUNCT.sub(" ", text)
    if strip_zero_width:
        text = _ZERO_WIDTH.sub("", text)
    text = text.lower()
    return _WS.sub(" ", text).strip()
