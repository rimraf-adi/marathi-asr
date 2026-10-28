from typing import List, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class SraVaaniAsr:
    """Model adapter for ARTPARK-IISc/SraVaani FastConformer-TDT ASR models."""

    def __init__(
        self,
        model_id: str = "ARTPARK-IISc/SraVaani-1.0",
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.name = model_id
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(self.device).eval()

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
        except Exception:
            self.processor = None

    def transcribe(self, audio: List[np.ndarray], sample_rate: int = 16000) -> List[str]:
        results = []
        for i in range(0, len(audio), self.batch_size):
            batch = audio[i : i + self.batch_size]
            if self.processor is not None:
                inputs = self.processor(
                    [a.astype(np.float32) for a in batch],
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                with torch.inference_mode():
                    logits = self.model.generate(**inputs)
                    texts = self.processor.batch_decode(logits)
                    results.extend(texts)
            else:
                # Direct transcribe fallback if supported by model class
                if hasattr(self.model, "transcribe_audio_arrays"):
                    texts = self.model.transcribe_audio_arrays(batch, sample_rate=sample_rate)
                    results.extend(texts)
                else:
                    results.extend([""] * len(batch))
        return results
