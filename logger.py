import csv
import os
from datetime import datetime, timedelta

from config import LOG_DIR, LOG_RETENTION_DAYS, ENABLE_LOGS

# ================== LOGGING ==================
current_user_log = None
current_ai_log = None
current_mood_log = None
current_date_str = None


def get_daily_log_files():
    """Ensure daily log files exist and return their paths."""
    global current_user_log, current_ai_log, current_mood_log, current_date_str
    today_str = datetime.now().strftime("%Y%m%d")

    if current_date_str != today_str or current_user_log is None:
        current_date_str = today_str

        os.makedirs(LOG_DIR, exist_ok=True)

        # User log
        current_user_log = os.path.join(LOG_DIR, f"user_log_{today_str}.csv")
        if not os.path.exists(current_user_log):
            with open(current_user_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_transcript", "user_emotion"])

        # AI log
        current_ai_log = os.path.join(LOG_DIR, f"ai_log_{today_str}.csv")
        if not os.path.exists(current_ai_log):
            with open(current_ai_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "ai_reply"])

    return current_user_log, current_ai_log, current_mood_log


def log_user_transcript(transcript, emotion_str: str = ""):
    if not ENABLE_LOGS:
        return
    log_file, _, _ = get_daily_log_files()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, transcript, emotion_str])


def log_ai_reply(reply):
    if not ENABLE_LOGS:
        return
    _, log_file, _ = get_daily_log_files()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, reply])


def log_mood_prediction(happy, sad, angry, neutral, fear, surprise, calm, disgust):
    if not ENABLE_LOGS:
        return
    _, _, log_file = get_daily_log_files()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            happy, sad, angry, neutral,
            fear, surprise, calm, disgust
        ])


def cleanup_old_logs(days=LOG_RETENTION_DAYS):
    cutoff = datetime.now() - timedelta(days=days)
    for filename in os.listdir(LOG_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(LOG_DIR, filename)
            file_ctime = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_ctime < cutoff:
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted old log: {filename}")
                except Exception as e:
                    print(f"⚠️ Could not delete {filename}: {e}")


def initialize_logging():
    """Initialize logging directory."""
    os.makedirs(LOG_DIR, exist_ok=True)
