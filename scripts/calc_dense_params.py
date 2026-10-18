import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from model.asr_model import StreamingASRModel

def test_configs():
    # Candidates to reach ~130.8M parameters with a dense Conformer:
    candidates = [
        # Option A: Deeper model (num_layers=24 or 22, d_model=512)
        {"d_model": 512, "num_layers": 24, "n_heads": 8, "ffn_expansion": 4, "desc": "24 layers, d=512, ffn=4 (Deep)"},
        {"d_model": 512, "num_layers": 25, "n_heads": 8, "ffn_expansion": 4, "desc": "25 layers, d=512, ffn=4"},
        {"d_model": 512, "num_layers": 26, "n_heads": 8, "ffn_expansion": 4, "desc": "26 layers, d=512, ffn=4"},
        # Option B: Wider FFN (num_layers=12, d_model=512, wider FFN)
        {"d_model": 512, "num_layers": 12, "n_heads": 8, "ffn_expansion": 10, "desc": "12 layers, d=512, ffn=10 (Wide FFN)"},
        {"d_model": 512, "num_layers": 12, "n_heads": 8, "ffn_expansion": 11, "desc": "12 layers, d=512, ffn=11 (Wide FFN)"},
        # Option C: Larger d_model (d_model=768, num_layers=16 or 14)
        {"d_model": 768, "num_layers": 14, "n_heads": 12, "ffn_expansion": 4, "desc": "14 layers, d=768, ffn=4"},
        {"d_model": 768, "num_layers": 15, "n_heads": 12, "ffn_expansion": 4, "desc": "15 layers, d=768, ffn=4"},
    ]

    for c in candidates:
        m = StreamingASRModel(
            feat_dim=80,
            d_model=c["d_model"],
            num_layers=c["num_layers"],
            n_heads=c["n_heads"],
            conv_kernel_size=31,
            ffn_expansion=c["ffn_expansion"],
            dropout=0.1,
            exit_layers=[c["num_layers"]],
            enable_reconstruction_head=False,
            vocab_size=105,
        )
        tot = sum(p.numel() for p in m.parameters())
        print(f"{c['desc']}: {tot / 1e6:.2f}M parameters ({tot:,} params)")

if __name__ == "__main__":
    test_configs()
