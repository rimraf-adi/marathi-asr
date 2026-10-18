import json
import urllib.request
import urllib.parse
from pathlib import Path

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")

def send_4am_update():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bot_token = cfg["bot_token"]
    chat_id = cfg["chat_id"]

    message = """🌟 <b>4:00 AM MILESTONE UPDATE: 130M DENSE BASELINE RUNNING</b>

🏆 <b>MoE Pipeline (scaled-conformer512-moe-3stage) 100% Complete!</b>
• All 4 stages finished (Pretraining $\to$ MoE Adaptation $\to$ Calibration $\to$ Frozen Benchmark).
• <b>Frozen RESPIN Test Results (Layer 12, Beam Width 16):</b>
  - <b>D3 (Standard Marathi):</b> <b>5.78% CER</b> | 21.11% WER
  - <b>D4 (Varhadi / Vidarbha):</b> <b>6.85% CER</b> | 27.26% WER
  - <b>D2 (Ahirani / Khandesh):</b> <b>7.16% CER</b> | 29.58% WER
  - <b>D1 (Malvani / Konkan):</b> <b>9.12% CER</b> | 35.35% WER
  - <b>Overall Aggregate:</b> <b>7.26% CER</b> | <b>28.42% WER</b> (0.0025x RTF)

⚡ <b>New Active Run: 130M Dense Conformer Baseline</b>
• <b>Run Name:</b> <code>scaled-conformer512-dense-3stage</code>
• <b>Model Capacity:</b> <b>131.14M parameters</b> (parameter-matched to MoE's total capacity: 12 layers, d=512, ffn_exp=8).
• <b>Current Stage:</b> <b>Stage 1 Pretraining Active</b> (capped at <b>30,000 steps</b> as requested).
• <b>Process:</b> Active on GPU (PID: <code>28336</code>).
• <b>Data Streaming:</b> Streaming verified clean from Shrutilipi and Vaani (>1,700 hours Marathi speech).

💾 <b>Storage & Health:</b>
• <b>D: Drive Free Space:</b> <b>97.84 GB</b> stable.
• <b>Watchdog:</b> Zero deadlocks detected.

🚀 <b>Next Phase:</b>
• Sequential orchestrator is managing Stage 1 (30k) $\to$ Stage 2 Dense Adaptation (80k) $\to$ Stage 3 Calibration (40k).
• Upon completion, the <code>inference/moe_vs_notmoe/</code> suite will automatically evaluate both 130M models side-by-side to test the <b>Negative Transfer Hypothesis</b>!

<i>Next scheduled status report at 6:00 AM.</i>"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram API success:", res.get("ok"))

if __name__ == "__main__":
    send_4am_update()
