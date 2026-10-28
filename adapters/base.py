from typing import List, Protocol
import numpy as np


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: List[np.ndarray], sample_rate: int) -> List[str]:
        """Transcribe a batch of raw 1D float32 audio arrays sampled at sample_rate."""
        ...
