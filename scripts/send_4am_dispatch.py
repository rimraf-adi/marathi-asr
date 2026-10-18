import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("d:/marathi-asr/scripts/telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """🎙 <b>Marathi Conformer-512 MoE — 4:00 AM Progress Dispatch</b>

<b>Stage 2: 3-Dialect MoE Adaptation (Phase 4B: Dynamic Soft Routing)</b>
• <b>Progress:</b> <b>Step 28,000 / 80,000</b> (<b>35.0% completed!</b>). Over a third of the entire MoE adaptation curriculum has completed!
• <b>Active Phase:</b> Phase 4B Dynamic MoE Routing — Router gates are dynamically balancing acoustic representations across Puneri, Nagpuri, and Marathwada dialect modules.

<b>Major Loss & Convergence Milestone:</b>
• <b>Sub-1.0 CTC Loss Achieved!</b> CTC Loss has officially crashed below the 1.0 barrier, dropping down to <b>0.8976 – 0.9481</b> (down from 1.0990 at 2 AM, 1.2191 at midnight, 1.3959 at 10 PM, and 11.85 initial).
• <b>Character Error Rate (CER):</b> Steady at <b>20.7% – 24.8%</b> on colloquial dialect audio.
• <b>Qualitative Audio Alignment:</b>
   - <i>Ref :</i> <code>मोजमापनाची साधने किती प्रकारची असू शकतात ?</code>
   - <i>Pred:</i> <code>मुतमापणाची साधणी किती प्रकारची असू शकतात ?</code>
   Full multi-word clause (<i>"किती प्रकारची असू शकतात ?"</i>) transcribing with 100% exact character alignment!
   - <i>Ref :</i> <code>पिकाच्या वाढीसाठी युरिया वापरू का ?</code>
   - <i>Pred:</i> <code>पिकाा वाढीसाठी योजिया वापरू का ?</code>

<b>Storage & Infrastructure Status:</b>
• <b>Auto-Rotation Live:</b> Step 28,000 committed; Step 25,000 automatically pruned. Milestones <code>step_10000.pt</code> and <code>step_20000.pt</code> permanently locked. Active rolling window at <code>26000.pt</code>, <code>27000.pt</code>, <code>28000.pt</code> + <code>best_model.pt</code>.
• <b>Disk Space:</b> Rock-solid at <b>10.51 GB</b> free on Drive D:.
• <b>GPU Vitals:</b> NVIDIA RTX A5000 running exceptionally cool at <b>61°C</b>, 14.3 GB allocated, 81.1W.
• <b>Deadlock Watchdog:</b> 85 silent checks completed, zero deadlocks or stalls.

<i>Next scheduled manual progress dispatch in 2 hours (6:00 AM).</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram 4 AM Dispatch Result:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
