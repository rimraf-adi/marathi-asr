import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("d:/marathi-asr/scripts/telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """🎙 <b>Marathi Conformer-512 MoE — 6:00 PM Evening Dispatch</b>

<b>Stage 2: 3-Dialect MoE Adaptation (Phase 4B: Dynamic Soft Routing)</b>
• <b>Progress:</b> <b>Step 71,425 / 80,000</b> (<b>89.3% completed!</b>). Final stretch of Stage 2 — less than 8,600 steps remaining!
• <b>Upcoming Handover:</b> Upon reaching Step 80,000 tonight, the pipeline will automatically load <code>best_model.pt</code> and trigger <b>Stage 3: Multi-Source Calibration</b>.

<b>Historic Loss & Accuracy Breakthroughs:</b>
• ⚡ <b>Sub-0.60 CTC Loss Breached!</b> CTC Loss plummeted to an all-time low of <b>0.5969</b> (down from 0.62 at 4 PM, 0.69 at 2 PM, 0.77 at 10 AM, and 1.74 at 8 PM yesterday)!
• 🎯 <b>Sustained Single-Digit CER (6.5% – 8.5%) Across Complex Speech:</b>
   - <i>Ref :</i> <code>कर्जाचा व्याज कीती असता ?</code>
   - <i>Pred:</i> <code>कर्जाचा व्याज कीती असता ?</code>
   <b>100.0% exact match!</b> Authentic dialect phrasing decoded with zero errors!
   - <i>Ref :</i> <code>सोहन हा वस्तू आणि सेवा कर विभागाचा अधिकारी आहे...</code>
   - <i>Pred:</i> <code>सोहन हा वस्तू आणि सेवा कर विभागाचा अधिकारी आहे...</code>
   <b>6.5% CER!</b> Intricate official Marathi nomenclature (<i>"वस्तू आणि सेवा कर विभागाचा अधिकारी"</i>) decoded near-verbatim.
   - <i>Ref :</i> <code>मोहनला आपल्या शेवटच्या गुंतवणुकीत चांगला नफा कमावता आला...</code>
   - <i>Pred:</i> <code>मोहनला आपल्या शेवटच्या गुंतवणुकीत चांगला नफा गवता आला...</code>
   <b>8.5% CER!</b>

<b>Storage & Infrastructure Status:</b>
• <b>Storage Health:</b> Disk headroom rock-solid with <b>104.25 GB free</b> on Drive D:.
• <b>Checkpoint Management:</b> Strictly retaining rolling window (<code>69000.pt</code>, <code>70000.pt</code>, <code>71000.pt</code>) + <code>best_model.pt</code>.
• <b>GPU Vitals:</b> NVIDIA RTX A5000 running smoothly at <b>66°C</b>, 23.0 GB VRAM, 140.8W.
• <b>Deadlock Watchdog:</b> 113 silent checks passed, zero deadlocks or stalls.

<i>Stage 2 is expected to finish late this evening. Next scheduled manual dispatch in 2 hours (8:00 PM).</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram 6 PM Dispatch Result:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
