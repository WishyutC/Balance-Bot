from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import csv
import os
import uvicorn

from mood_parser import get_average_mood_from_csv

app = FastAPI()

# --------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://balancebot-app.pkongkha.org"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Paths ----------------
LOGS_DIR = "logs"                  # CSV logs
AUDIO_DIR = "audio_pcm"      # WAV files

# Serve WAVs as static files
# Serve WAVs at /output_realtime/...
app.mount("/audio_pcm", StaticFiles(directory=AUDIO_DIR), name="audio_pcm")




# --------------- Models ----------------
class Conversation(BaseModel):
    id: int
    type: str
    message: str
    timestamp: datetime
    audio_url: Optional[str] = None
    mood_score: Optional[float] = None

class DashboardData(BaseModel):
    current_mood: float
    mood_trends: List[dict]
    conversation_counts: List[dict]
    conversations: List[Conversation]

# --------------- Helpers ----------------
async def parse_latest_log():
    files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".csv")]
    if not files:
        return []

    files.sort(key=lambda f: os.path.getmtime(os.path.join(LOGS_DIR, f)), reverse=True)
    newest_file = os.path.join(LOGS_DIR, files[0])

    conversations = []
    idx = 1
    try:
        with open(newest_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if not row or not row[0].strip():
                    continue
                timestamp_str = row[0].strip()
                message = row[1].strip() if len(row) > 1 else ""

                # parse timestamp from log
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                # expected wav filename format
                wav_filename = f"B_bot_reply{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.wav"
                wav_path = os.path.join(AUDIO_DIR, wav_filename)

                audio_url = f"{wav_filename}" if os.path.exists(wav_path) else None

                conversations.append(
                    Conversation(
                        id=idx,
                        type="bot",
                        message=message,
                        timestamp=timestamp,
                        audio_url=audio_url
                    )
                )
                idx += 1
    except Exception as e:
        print(f"Error reading log: {e}")

    return conversations

# --------------- API ----------------
@app.get("/generate", response_model=DashboardData)
async def get_dashboard_data():
    conversations = await parse_latest_log()

    # For simplicity, using dummy data for mood trends and conversation counts
    current_mood = get_average_mood_from_csv(os.path.join(LOGS_DIR, f"user_log_{datetime.now().strftime('%Y%m%d')}.csv"))
    mood_trends = [
        {"day":"Mon","mood":6},
        {"day":"Tue","mood":7},
        {"day":"Wed","mood":8},
        {"day":"Thu","mood":5},
        {"day":"Fri","mood":7},
        {"day":"Sat","mood":6.5},
        {"day":"Sun","mood":7.5}
    ]
    conversation_counts = [
        {"day":"Mon","count":5},
        {"day":"Tue","count":3},
        {"day":"Wed","count":7},
        {"day":"Thu","count":4},
        {"day":"Fri","count":6},
        {"day":"Sat","count":2},
        {"day":"Sun","count":5}
    ]

    return DashboardData(
        current_mood=current_mood,
        mood_trends=mood_trends,
        conversation_counts=conversation_counts,
        conversations=conversations
    )

# --------------- Run server ----------------
if __name__ == "__main__":
    uvicorn.run("pi_api:app", host="0.0.0.0", port=3000, reload=True)
