import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """🚀 <b>Marathi Conformer-512 MoE — Stage 2 (3-Dialect MoE) KICKOFF!</b>

<b>Stage 1 Pretraining Verified & Completed:</b>
• <b>Duration:</b> 30.15 Hours Continuous (1,809 mins)
• <b>Audio Volume:</b> 1,763.9 Hours across 1,160,977 unique utterances
• <b>Convergence:</b> Masked L1 dipped to <b>0.3545</b>, Peak SNR <b>+21.18 dB</b>
• <b>Checkpoint:</b> <code>best_model.pt</code> (d_model=512, 130.82M parameters) loaded

<b>Stage 2: 3-Dialect MoE Adaptation Commenced:</b>
• <b>Target Steps:</b> <b>80,000 steps</b>
• <b>Dataset:</b> IISc RESPIN Clean Train Set (95% split, cached speaker partition)
• <b>Architecture:</b> Conformer Layers 5–12 upcycled to 3 Dialect Experts (74.3% active / 25.7% dormant params)
• <b>Curriculum Phases:</b>
   1. <b>Phase 0 (Steps 1 – 3,000):</b> CTC Warmup (Backbone calibration before routing)
   2. <b>Phase 4A (Steps 3,001 – 18,000):</b> Metadata-Guided Hard Routing (Puneri / Nagpuri / Marathwada)
   3. <b>Phase 4B (Steps 18,001 – 80,000):</b> Dynamic Soft Top-1 Routing
• <b>Optimization:</b> Batch Size 64 (accum=2, effective batch=128), Differential LR (backbone 1/10 LR, MoE+CTC full LR 1.5e-4)

<b>Hardware Status:</b>
• <b>GPU:</b> NVIDIA RTX A5000 running at <b>100% compute load</b> (109 W, 58°C)

<i>Stage 2 MoE is actively training. Periodic reports continue every 2 hours; silent deadlock watchdog running every 30 minutes.</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Sent successfully:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
