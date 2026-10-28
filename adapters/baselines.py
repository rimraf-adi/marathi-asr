import os
import tempfile
from typing import List

import numpy as np
import soundfile as sf
import torch


class _PathAdapter:
    def _with_wavs(self, audio, sr):
        paths = []
        for a in audio:
            fd, p = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(p, np.asarray(a, dtype=np.float32), sr)
            paths.append(p)
        return paths


class SraVaaniBaseline(_PathAdapter):
    def __init__(self, repo_id="ARTPARK-IISc/SraVaani-1.0", device="cuda"):
        from inference.pipeline import SraVaaniModel
        self.name = repo_id
        self.m = SraVaaniModel(repo_id, token=os.environ.get("HF_TOKEN"), device=device)

    def transcribe(self, audio: List[np.ndarray], sample_rate: int = 16000) -> List[str]:
        paths = self._with_wavs(audio, sample_rate)
        try:
            return self.m.transcribe_paths(paths, batch_size=len(paths))
        finally:
            for p in paths:
                os.remove(p)


class IndicConformerBaseline(_PathAdapter):
    def __init__(self, repo_id="ai4bharat/indic-conformer-600m-multilingual", device="cuda"):
        from inference.pipeline import IndicConformerModel
        self.name = repo_id
        self.m = IndicConformerModel(repo_id, token=os.environ.get("HF_TOKEN"), device=device)
        self.m.model.to(self.m.device)

    def transcribe(self, audio: List[np.ndarray], sample_rate: int = 16000) -> List[str]:
        paths = self._with_wavs(audio, sample_rate)
        try:
            return [self.m.transcribe_single(p) for p in paths]
        finally:
            for p in paths:
                os.remove(p)
