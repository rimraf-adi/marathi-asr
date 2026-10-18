import json
import urllib.request
import urllib.parse
from pathlib import Path

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")

def send_midnight_update():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bot_token = cfg["bot_token"]
    chat_id = cfg["chat_id"]

    message = """🌙 <b>12:00 AM MIDNIGHT PROGRESS UPDATE: STAGE 3 CALIBRATION (32.9%)</b>

🎯 <b>Stage 3 (Multi-Source Calibration & Sequence Polishing):</b>
• <b>Active Step:</b> <b>13,150 / 40,000</b> (32.9% completed — nearly one-third done!)
• <b>CTC Loss:</b> Converging between <b>0.7085 – 0.8223</b> across diverse acoustic channels
• <b>Throughput:</b> Sustained at <b>356 – 466 audio-sec/s</b>
• <b>Data Mixture:</b> Seamless 4-stream blending (held-out RESPIN dialects, Kathbath crowdsourced, OpenSLR64 read speech, and pretrain stream)
• <b>Learning Rate:</b> 2.96e-05 (following cosine decay schedule)

💾 <b>Checkpoints & Disk Headroom:</b>
• <b>Latest Checkpoint:</b> <code>stage3_final_alignment_step_13000.pt</code>
• <b>Strict Auto-Rotation:</b> Preserving only latest 3 step checkpoints (12000, 12500, 13000) + <code>best_model.pt</code>
• <b>Disk Free Space (D:):</b> <b>97.84 GB</b> stable (zero bloat)

🖥️ <b>Hardware & System Health:</b>
• <b>Process:</b> Active (PID: <code>14444</code>) | Watchdog verified: Zero deadlocks detected.
• <b>GPU:</b> RTX A5000 @ 84°C | 198.1W | 24.03 GB / 24.56 GB VRAM (97.8% utilization).

⏳ <b>Automated Pipeline Queued:</b>
• <code>schedule_moe_vs_dense.py</code> is standing by in background.
• As soon as Stage 3 & final benchmark finish, it will automatically initiate the <b>130M Dense Conformer Baseline</b> and run the comparative benchmark on the frozen RESPIN test set!

<i>Next scheduled report at 2:00 AM.</i>"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram API success:", res.get("ok"))

if __name__ == "__main__":
    send_midnight_update()
