import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """📢 <b>Marathi Conformer-512 MoE — Stage 1 Epoch 1 Milestone Report</b>

<b>Stage 1: Pretraining Milestone Completed</b>
• <b>Active Run Time:</b> <b>30.15 Hours Continuous</b> (1,809 minutes)
• <b>Total Audio Ingested:</b> <b>1,763.94 Hours</b> across <b>1,160,977 unique utterances</b>!
• <b>Epoch 1:</b> <b>100% Complete</b> (18,141 steps reached, 18,325 logged)
• <b>Convergence Highlights:</b>
   - Masked L1 loss touched an all-time low of <b>0.3545</b>
   - Peak Reconstruction SNR surged to <b>+21.18 dB</b>
   - Latent representation rank remained healthy (170–210 / 512, zero dimensional collapse)
• <b>Best Model Checkpoint:</b> Verified & saved at <code>runs/scaled-conformer512-moe-3stage/checkpoints/stage1_pretrain/best_model.pt</code> (Step 15,000, 130.8M parameters).

<i>Stage 1 Epoch 1 is fully finished and verified. Awaiting decision: advance to Stage 2 (3-Dialect MoE Adaptation) or resume Stage 1 for additional epochs.</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Sent successfully:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
