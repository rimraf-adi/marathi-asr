import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np

# Ensure root directory is in sys.path
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from moe.upcycling import load_moe_model
from moe.respin_dataset import build_or_load_speaker_split, collate_moe_batch
from data_utils.tokenizer import MarathiTokenizer

def extract_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting on {device}")
    
    ckpt_path = "runs/no-splitformer-moe-only/checkpoints/stage2_moe/best_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found!")
        from model.asr_model import StreamingASRModel
        from moe.moe_layer import SparseMoELayer
        model = StreamingASRModel(feat_dim=80, d_model=256, num_layers=12, n_heads=4, conv_kernel_size=31, ffn_expansion=4, dropout=0.1, exit_layers=[12], enable_reconstruction_head=False, vocab_size=105)
        for layer_idx in [4,5,6,7,8,9,10,11]:
            model.encoder.layers[layer_idx].ffn2 = SparseMoELayer(d_model=256, expansion_factor=4, num_experts=3, dropout=0.1)
        model.to(device)
    else:
        model = load_moe_model(ckpt_path, device=device)
    model.eval()

    # Monkeypatch to capture router probs
    layer_to_track = 8
    captured_probs = []
    
    target_moe_layer = model.encoder.layers[layer_to_track].ffn2
    original_forward = target_moe_layer.forward
    
    def custom_forward(self, x, dialect_idx=None, use_hard_routing=False):
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        self._last_router_probs = router_probs.detach().cpu().numpy()
        return original_forward(x, dialect_idx, use_hard_routing)
        
    target_moe_layer.forward = custom_forward.__get__(target_moe_layer, type(target_moe_layer))

    tokenizer = MarathiTokenizer("data_utils/vocab.json")
    _, calib_utts = build_or_load_speaker_split()
    
    selected = {}
    for utt in calib_utts:
        d = utt["dialect"]
        if d in ["D1", "D2", "D4"] and d not in selected:
            if utt["duration"] < 5.0 and utt["duration"] > 2.0:
                selected[d] = utt
        if len(selected) == 3:
            break
            
    print("Selected examples:", [v["uid"] for v in selected.values()])
    
    results = {}
    for d, utt in selected.items():
        batch = collate_moe_batch([utt], tokenizer, augment=False)
        if batch is None:
            continue
            
        audio = batch["audio"].to(device)
        audio_lens = batch["audio_lengths"].to(device)
        
        with torch.no_grad():
            model.forward_ctc(audio)
            
        probs = target_moe_layer._last_router_probs[0] # (T, 3)
        T_actual = int(audio_lens[0].item() // 4)
        probs = probs[:T_actual]
        
        indices = np.linspace(0, len(probs)-1, 50, dtype=int)
        sampled_probs = probs[indices].tolist()
        
        results[d] = {
            "text": utt["text"],
            "dialect": d,
            "probs": sampled_probs
        }
        
    with open("illustrations/moe_activations.json", "w") as f:
        json.dump(results, f)
    print("Extraction complete. Saved to illustrations/moe_activations.json")

if __name__ == "__main__":
    extract_data()
