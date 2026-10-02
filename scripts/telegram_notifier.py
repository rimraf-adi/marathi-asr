import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def get_latest_metrics(run_dir):
    csv_path = Path(run_dir) / "metrics.csv"
    if not csv_path.exists():
        return "Metrics CSV not found yet."
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return "Metrics CSV is empty."
        latest = df.iloc[-1]
        
        msg = f"<b>Step:</b> {latest['step']}\n"
        
        # Format metrics conditionally based on what's available
        if "phase" in latest:
            msg += f"<b>Phase:</b> {latest['phase']}\n"
            
        for key in ["ctc_loss", "aux_loss", "cer", "grad_norm", "throughput_aud_s_per_s"]:
            if key in latest:
                msg += f"<b>{key}:</b> {latest[key]:.4f}\n"
                
        # For GRPO stages
        for key in ["reward_wer", "reward_dialect", "advantage", "policy_loss", "value_loss", "entropy_loss", "kl_div"]:
            if key in latest:
                msg += f"<b>{key}:</b> {latest[key]:.4f}\n"
                
        return msg
    except Exception as e:
        return f"Error reading metrics: {e}"

def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID missing in .env")
        return
        
    print("Checking active runs for telemetry...")
    
    # Priority order of runs
    active_runs = [
        ("Stage 5B (OpenSLR Polish)", "runs/stage5_openslr_polish"),
        ("Stage 4B (Kathbath GRPO)", "runs/moe_grpo_kathbath"),
        ("Stage 4A (RESPIN Hard Routing)", "runs/moe_hard_routing")
    ]
    
    status_msg = "🚀 <b>Marathi ASR Pipeline Update</b>\n\n"
    found_active = False
    
    for stage_name, run_dir in active_runs:
        if Path(run_dir).exists():
            metrics_str = get_latest_metrics(run_dir)
            status_msg += f"🔥 <b>Current Stage:</b> {stage_name}\n"
            status_msg += f"{metrics_str}\n"
            found_active = True
            break # Only report the furthest active stage
            
    if not found_active:
        status_msg += "No active training stages found. Pipeline might be finished or hasn't started logging!"
        
    send_telegram_message(token, chat_id, status_msg)
    print("Notification sent!")

if __name__ == "__main__":
    main()
