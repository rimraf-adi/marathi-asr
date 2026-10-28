from typing import List

import numpy as np
import torch

from data_utils.tokenizer import MarathiTokenizer, sanitize_transcript


class OurConformer:
    """Greedy-CTC adapter for our 130M dense / MoE Conformer-512 training checkpoints.

    Loads straight from a training checkpoint (runs/.../best_model.pt), never from exported copies,
    and fails if the dense loader would silently drop weights.
    """

    def __init__(self, checkpoint_path: str, name: str, vocab_path: str = "data_utils/vocab.json", device: str = "cuda"):
        from model.asr_model import StreamingASRModel
        from moe.upcycling import load_moe_model

        self.name = name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.is_moe = any("ffn2.experts" in k for k in sd)
        if self.is_moe:
            self.model = load_moe_model(checkpoint_path, device=self.device)
        else:
            cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
            d_model = cfg.get("d_model", 256)
            self.model = StreamingASRModel(
                feat_dim=80, d_model=d_model, num_layers=cfg.get("num_layers", 12),
                n_heads=cfg.get("n_heads", 8 if d_model == 512 else 4),
                conv_kernel_size=31, ffn_expansion=cfg.get("ffn_expansion", 4), dropout=0.0,
                exit_layers=[12], enable_reconstruction_head=False, vocab_size=105,
            ).to(self.device)
            state = {k.replace("backbone.", "", 1) if k.startswith("backbone.") else k: v for k, v in sd.items()}
            res = self.model.load_state_dict(state, strict=False)
            lost = [k for k in res.missing_keys if "ctc" in k or "encoder" in k]
            if lost:
                raise RuntimeError(f"dense checkpoint left {len(lost)} encoder/ctc weights unloaded, e.g. {lost[:3]}")
        self.model.eval()
        self.tokenizer = MarathiTokenizer(vocab_path)
        self.n_params_m = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)

    @torch.no_grad()
    def transcribe(self, audio: List[np.ndarray], sample_rate: int = 16000) -> List[str]:
        lens = [len(a) for a in audio]
        padded = torch.zeros(len(audio), max(lens), dtype=torch.float32, device=self.device)
        for i, a in enumerate(audio):
            padded[i, : len(a)] = torch.from_numpy(np.asarray(a, dtype=np.float32))
        log_probs = self.model.forward_ctc(padded, chunk_size=None)["log_probs"]
        out = []
        for b in range(len(audio)):
            # frames beyond this utterance are padding; 4x conv + 2x... derive the cap from the ratio
            T = log_probs.shape[1]
            keep = max(1, int(round(T * lens[b] / max(lens))))
            ids = log_probs[b, :keep].argmax(-1).cpu().tolist()
            out.append(sanitize_transcript(self.tokenizer.ctc_decode(ids)))
        return out
