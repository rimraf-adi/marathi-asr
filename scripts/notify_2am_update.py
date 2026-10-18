import json
import urllib.request
import urllib.parse
from pathlib import Path

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")

def send_2am_update():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bot_token = cfg["bot_token"]
    chat_id = cfg["chat_id"]

    message = """⚡ <b>2:00 AM PROGRESS UPDATE: STAGE 3 CALIBRATION (68.5%)</b>

🎯 <b>Stage 3 (Multi-Source Calibration & Final Alignment):</b>
• <b>Active Step:</b> <b>27,400 / 40,000</b> (<b>68.5% complete</b> — over two-thirds done!)
• <b>Remaining Steps:</b> ~12,600 steps (~1.5 hours remaining)
• <b>CTC Loss:</b> Converging at <b>0.7188 – 0.7329</b> across all 4 blended domains
• <b>Sample CER:</b> Down to <b>17.1%</b> on noisy colloquial speech
• <b>Throughput:</b> Fast and steady at <b>375 – 450 audio-sec/s</b>
• <b>Learning Rate:</b> 2.80e-05 (smooth cosine schedule)

💾 <b>Checkpoints & Storage Headroom:</b>
• <b>Latest Checkpoint:</b> <code>stage3_final_alignment_step_27000.pt</code>
• <b>Rotation Active:</b> Retaining strictly latest 3 checkpoints (26000, 26500, 27000) + <code>best_model.pt</code>
• <b>D: Drive Free Space:</b> <b>97.84 GB</b> stable

🖥️ <b>Hardware Health & Stability:</b>
• <b>Process:</b> PID <code>14444</code> | Deadlock watchdog verified: Zero deadlocks detected.
• <b>GPU:</b> RTX A5000 @ 78°C | 191.6W | 24.03 GB / 24.56 GB VRAM.

🚀 <b>Next in Queue:</b>
• The sequential orchestrator (<code>schedule_moe_vs_dense.py</code>) is active in background.
• As soon as Stage 3 finishes, it will automatically initiate the <b>130M Dense Conformer Baseline</b> (30k pretraining + 80k adaptation + 40k calibration) and run the comparative benchmark on the frozen RESPIN test set!

<i>Next scheduled report at 4:00 AM.</i>"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram API success:", res.get("ok"))

if __name__ == "__main__":
    send_2am_update()
