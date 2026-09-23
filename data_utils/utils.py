import math
try:
    import editdistance
except ImportError:
    editdistance = None
import jiwer
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate (CER)."""
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    if editdistance is not None:
        return float(editdistance.eval(ref, hyp)) / len(ref)
    return float(jiwer.cer(ref, hyp))

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
