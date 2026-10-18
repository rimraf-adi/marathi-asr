import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("d:/marathi-asr/scripts/telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """🎙 <b>Marathi Conformer-512 MoE — 12:00 AM Midnight Progress Dispatch</b>

<b>Stage 2: 3-Dialect MoE Adaptation</b>
• <b>Active Phase:</b> Phase 4A (Hard Routing) — Final stretch!
• <b>Step Progress:</b> <b>15,750 / 80,000</b> (~19.7% of Stage 2 completed).
• <b>Milestone Approaching:</b> Only ~2,250 steps remaining until the major handover to <b>Phase 4B (Dynamic Soft / Top-1 Routing)</b> at Step 18,000, where router gates will autonomously steer dialect representations.

<b>Model Convergence & Alignment:</b>
• <b>CTC Loss:</b> Converged further to <b>1.2191</b> (down from 1.3959 at 10 PM and 1.7410 at 8 PM).
• <b>Greedy CER:</b> Dropped to <b>22.5% – 22.8%</b> on colloquial dialect Marathi utterances.
• <b>Qualitative Audio Alignment:</b>
   - <i>Ref :</i> <code>ऊस कापणीकरिता कोणते यंत्र वापरले जाते ? बाजारात त्याची किंमत किती ?</code>
   - <i>Pred:</i> <code>ऊस कापणी करिता कोणते यंत्र वापरले जाके बाजारात त्याचीकिंम किती ?</code>
   Complex agricultural domain vocabulary and Devnagari compound letters (जोडाक्षरे) are decoding with remarkable phonetic fidelity.
• <b>Throughput:</b> Surging up to <b>160 – 276 audio-sec/sec</b>.

<b>Storage & Hardware Health:</b>
• <b>Checkpoints:</b> Milestone <code>step_10000.pt</code> safely locked; active rolling window at <code>step_13000.pt</code>, <code>14000.pt</code>, <code>15000.pt</code> + <code>best_model.pt</code>.
• <b>Storage Stability:</b> Free disk space rock-solid at <b>11.98 GB</b> with zero leakage.
• <b>GPU:</b> NVIDIA RTX A5000 @ <b>67°C</b>, 65% utilization, 22.2 GB / 24.5 GB, 101.2W.
• <b>Watchdog:</b> 77 silent checks passed, zero deadlocks.

<i>Next scheduled manual progress dispatch in 2 hours (2:00 AM).</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram Midnight Dispatch Result:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
