import json
import urllib.request
import urllib.parse
from pathlib import Path

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")

def send_update():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bot_token = cfg["bot_token"]
    chat_id = cfg["chat_id"]

    message = """📈 <b>10:00 PM PROGRESS UPDATE: STAGE 3 CALIBRATION (10.4%)</b>

🎯 <b>Stage 3 (Multi-Source Calibration) Status:</b>
• <b>Active Step:</b> <b>4,175 / 40,000</b> (10.4% completed in ~1h 18m)
• <b>CTC Loss:</b> Down to <b>0.7168</b> (starting from ~0.97 at step 0)
• <b>Throughput:</b> <b>333 – 400 audio-sec/s</b>
• <b>Corpora Blending:</b> Ingesting dynamic mix of RESPIN 5% held-out, Kathbath colloquial, OpenSLR64 clean read speech, and pretrain stream.
• <b>Learning Rate:</b> 3.00e-05 (cosine schedule targeting 5.0e-06)

💾 <b>Checkpoints & Retention:</b>
• <b>Latest Checkpoint:</b> <code>stage3_final_alignment_step_4000.pt</code>
• <b>Strict Rotation:</b> Only last 3 step checkpoints (3000, 3500, 4000) + <code>best_model.pt</code> are kept on disk.
• <b>Disk Free Space (D:):</b> <b>97.84 GB</b> stable (healthy headroom).

🖥️ <b>Hardware & Runtime Health:</b>
• <b>Process:</b> Active (PID: <code>14444</code>) | No deadlocks detected.
• <b>GPU:</b> RTX A5000 @ 77°C | 168.7W | 23.98 GB / 24.56 GB VRAM (97.6% utilized).

<i>Next scheduled status report will be dispatched at 12:00 AM (midnight).</i>"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram API success:", res.get("ok"))

if __name__ == "__main__":
    send_update()
