import json
import urllib.request
import urllib.parse
from pathlib import Path

cfg_path = Path("telegram_config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

msg = """🎙 <b>Marathi Conformer-512 MoE — 6:00 PM Progress Dispatch</b>

<b>Stage 2: 3-Dialect MoE Adaptation Status</b>
• <b>Active Phase:</b> Phase 0 (CTC Warmup — Backbone & Linear Head Alignment)
• <b>Progress:</b> Stepping actively towards 80,000 steps
• <b>Optimizations & Fixes Applied:</b>
   - <b>Outlier Filtering:</b> Strict audio duration bounds (1.0s to 16.0s) active in collator, eliminating long-tail OOM outliers.
   - <b>Memory Efficiency:</b> Tuned to Batch Size 32, Grad Accum 4 (identical effective batch 128 utts/step).
   - <b>VRAM Headroom:</b> Operating at a lean <b>14.8 GB / 24.5 GB</b> (nearly 10 GB headroom!).
   - <b>Checkpoint Routine:</b> Fixed checkpoint metric signature (<code>cer</code>).
• <b>Convergence Trend:</b> CTC Loss collapsed rapidly from <b>11.85</b> down to <b>3.43</b> during initial warmup passes.
• <b>Throughput:</b> Steady at <b>140 – 175 aud-sec/s</b>.

<b>Compute & Hardware Health:</b>
• <b>GPU:</b> NVIDIA RTX A5000 running cool at 51°C
• <b>Deadlock Watchdog:</b> 65 iterations verified healthy, zero deadlocks

<i>Stage 2 MoE is actively training. Next scheduled dispatch in 2 hours (8:00 PM).</i>"""

url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": msg, "parse_mode": "HTML"}).encode("utf-8")
req = urllib.request.Request(url, data=data)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Sent successfully:", res.get("ok"))
except Exception as e:
    print("Error sending message:", e)
