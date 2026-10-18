import json
import urllib.request
import urllib.parse
from pathlib import Path

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")

def send_milestone():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bot_token = cfg["bot_token"]
    chat_id = cfg["chat_id"]

    message = """🚀 <b>MILESTONE UPDATE: STAGE 2 COMPLETED & STAGE 3 IN PROGRESS!</b>

🎉 <b>Stage 2 (MoE Adaptation) Finished Successfully!</b>
• <b>Total Steps:</b> 80,000 / 80,000 (100% complete)
• <b>Runtime:</b> 1,390.1 mins (~23.2 hours)
• <b>Convergence:</b> CTC Loss plummeted from 11.85 down to <b>0.5633</b>
• <b>Dialect Accuracy:</b> Single-digit greedy CER (<b>3.5% – 8.5%</b> on vernacular dialect samples)
• <b>MoE Routing:</b> Aux loss stabilized at ~8.01 with balanced gradient allocation across Puneri, Nagpuri, and Marathwada dialect experts
• <b>Checkpoint:</b> <code>checkpoints/stage2_moe/best_model.pt</code> saved and validated

⚡ <b>Stage 3 (Multi-Source Alignment & Calibration) Active:</b>
• <b>Status:</b> Automatically launched and currently training (PID: <code>14444</code>)
• <b>Progress:</b> <b>Step 2,050 / 40,000</b> (running for ~45 mins)
• <b>Current CTC Loss:</b> 0.8129 (rapidly dropping across combined domains)
• <b>Throughput:</b> <b>400 – 456 audio-sec/s</b> (ultra-fast 4-stream pipeline)
• <b>Corpora:</b> Blended RESPIN 5% held-out, Kathbath, OpenSLR64, and pretrain streams

🖥️ <b>Hardware & Storage Health:</b>
• <b>GPU:</b> NVIDIA RTX A5000 @ 72°C | 148W | 22.58 GB / 24.0 GB VRAM
• <b>D: Drive Free Space:</b> <b>97.84 GB free</b> (checkpoint auto-rotation active: keeping last 3 checkpoints + best_model.pt)
• <b>Recycle Bin:</b> 0 items (completely clear)

<i>Note: The 2-hour cron triggers on fixed even hours (20:00, 22:00). Since Stage 2 wrapped up at 20:42, this milestone alert was dispatched directly on completion. Standard 2-hour tracking remains active.</i>"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram API success:", res.get("ok"))

if __name__ == "__main__":
    send_milestone()
