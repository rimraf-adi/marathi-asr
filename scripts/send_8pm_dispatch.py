import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("d:/marathi-asr/scripts/telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """🎙 <b>Marathi Conformer-512 MoE — 8:00 PM Live Progress Dispatch</b>

<b>Stage 2: 3-Dialect MoE Adaptation (IISc RESPIN 95% Set)</b>
• <b>Active Phase:</b> Phase 4A (Metadata-Guided Hard Routing)
• <b>Step Count:</b> <b>5,500 / 80,000</b> (~6.9% of Stage 2 completed)
• <b>Active Progression:</b> Successfully transitioned out of Phase 0 Warmup (0–3,000 steps) and now advancing deep into Phase 4A Hard Routing (3,001–18,000 steps).

<b>Training Dynamics & Convergence:</b>
• <b>CTC Loss:</b> Converged steadily from initial <b>11.85</b> down to <b>1.7174</b> (recent trough at <b>1.6810</b>).
• <b>Greedy CER:</b> Dropped from >200% down to <b>37.5% – 46.4%</b> on conversational dialect inputs.
• <b>MoE Aux Loss:</b> Balanced at <b>8.01 – 8.11</b>, maintaining uniform expert utilization across Puneri, Nagpuri, and Marathwada blocks (layers 5–12).
• <b>Throughput:</b> Solid <b>110 – 180 audio-sec/sec</b>.
• <b>Checkpoints:</b> Checkpoint <code>stage2_moe_step_5000.pt</code> and latest <code>best_model.pt</code> (~1.58 GB) safely committed to disk.

<b>Hardware & Stability Telemetry:</b>
• <b>GPU:</b> NVIDIA RTX A5000 (24 GB) @ <b>67°C</b>, 122W draw.
• <b>Memory Allocation:</b> 21.8 GB / 24.5 GB stable, zero OOM spikes with duration bounds (1.0s–16.0s) active.
• <b>Watchdog Status:</b> Silent 30-min deadlock checks running nominally, zero frozen iterators.

<i>Model continues uninterrupted in Phase 4A towards the 18,000-step soft routing handover. Next dispatch in 2 hours (10:00 PM).</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram Dispatch Result:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
