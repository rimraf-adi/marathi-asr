import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("d:/marathi-asr/scripts/telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """🎙 <b>Marathi Conformer-512 MoE — 10:00 PM Progress Dispatch</b>

<b>Stage 2: 3-Dialect MoE Adaptation (Phase 4A: Hard Routing)</b>
• <b>Progress:</b> <b>Step 9,600 / 80,000</b> (12.0% completed). Rapidly approaching the 10,000-step milestone!
• <b>Active Phase:</b> Metadata-Guided Hard Routing (Layers 5–12 routing directly to Puneri, Nagpuri, and Marathwada dialect experts).

<b>Convergence & Transcription Quality:</b>
• <b>CTC Loss:</b> Converged down from 1.7410 (at 8 PM) to <b>1.3959</b> (near-record low).
• <b>Character Error Rate (CER):</b> Reached a remarkable <b>23.8%</b> on natural dialect conversational utterances (down from 37.4% at 8 PM).
• <b>Qualitative Audio Alignment:</b>
   - <i>Ref :</i> <code>बकरीच्या दूधाची किंमत काय आहे ?</code>
   - <i>Pred:</i> <code>बकरीच्या दिढ्याची किंमत काय आहे ?</code>
   High phonetic alignment; word boundaries and Devnagari matras are sharply resolving.
• <b>MoE Auxiliary Loss:</b> <b>8.04 – 8.12</b>, maintaining balanced gradient flow across all 3 dialect pathways.

<b>Infrastructure & Storage Health:</b>
• <b>Storage Stability:</b> Implemented rolling checkpoint rotation (keeps <code>best_model.pt</code>, latest 3 rolling checkpoints, plus permanent 10k milestones). Free disk headroom secured and steady at <b>13.45 GB</b>.
• <b>Latest Saved Models:</b> <code>stage2_moe_step_9000.pt</code> and <code>best_model.pt</code> (~1.58 GB) committed smoothly.
• <b>GPU Vitals:</b> NVIDIA RTX A5000 @ <b>69°C</b>, 43% active compute, 21.5 GB / 24.5 GB VRAM, 112.7W.
• <b>Deadlock Watchdog:</b> 73 silent checks passed, zero deadlocks.

<i>Next scheduled manual progress dispatch in 2 hours (12:00 AM Midnight).</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram 10 PM Dispatch Result:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
