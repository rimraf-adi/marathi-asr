from typing import List, Optional
import numpy as np
import torch
from transformers import pipeline


class HFAsr:
    def __init__(
        self,
        model_id: str,
        whisper: bool = False,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.name = model_id
        self.whisper = whisper
        self.batch_size = batch_size

        if device is None:
            dev = 0 if torch.cuda.is_available() else -1
        else:
            dev = int(device) if device.isdigit() else device

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=dev,
            torch_dtype=dtype,
        )

    def transcribe(self, audio: List[np.ndarray], sample_rate: int) -> List[str]:
        inputs = [{"raw": a.astype(np.float32), "sampling_rate": sample_rate} for a in audio]
        kwargs = {}
        if self.whisper:
            kwargs["generate_kwargs"] = {"language": "marathi", "task": "transcribe"}

        out = self.pipe(inputs, batch_size=self.batch_size, **kwargs)
        return [o.get("text", "") for o in out]
