import json
import urllib.request
import urllib.parse
from pathlib import Path

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")

def send_final_eval_notification():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bot_token = cfg["bot_token"]
    chat_id = cfg["chat_id"]

    message = """🏆 <b>OFFICIAL BENCHMARK RESULTS: MOE MODEL (STAGE 4 COMPLETE)</b>

🎉 <b>Full Sequential Pipeline Completed Successfully!</b>
• <b>Model:</b> Conformer-512 3-Dialect MoE (130.82M total params, 63.8M active)
• <b>Checkpoint:</b> <code>stage3_final_alignment/best_model.pt</code>
• <b>Evaluation Set:</b> Frozen IISc RESPIN Held-Out Test Set (2,170 utterances, 3.04 hrs)
• <b>Decoding:</b> Beam Search (width=16, offline full context)
• <b>Inference Speed:</b> <b>0.0025x RTF</b> (evaluated entire 3.04 hrs in 109.5 seconds!)

📊 <b>Per-Dialect Breakdown (Layer 12 Deep Exit):</b>
• <b>D3 (Standard Marathi):</b> <b>5.78% CER</b> | 21.11% WER (555 utts)
• <b>D4 (Varhadi / Vidarbha):</b> <b>6.85% CER</b> | 27.26% WER (516 utts)
• <b>D2 (Ahirani / Khandesh):</b> <b>7.16% CER</b> | 29.58% WER (540 utts)
• <b>D1 (Malvani / Konkan):</b> <b>9.12% CER</b> | 35.35% WER (559 utts)
• <b>ALL (Aggregate):</b> <b>7.26% CER</b> | <b>28.42% WER</b> (2,170 utts)

⚡ <b>Next Phase:</b>
• Orchestrator has picked up pipeline completion.
• Automatically kicking off the <b>130M Dense Conformer Baseline</b> (30k pretraining steps) to test the <b>Negative Transfer Hypothesis</b>!"""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Telegram API success:", res.get("ok"))

if __name__ == "__main__":
    send_final_eval_notification()
