from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import csv
import os
import base64
import uvicorn

app = FastAPI()

# --------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173/"],  # replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Paths ----------------
LOGS_DIR = "logs"                  # text logs CSV
AUDIO_DIR = "output_realtime"      # WAV files

# --------------- Models ----------------
class Conversation(BaseModel):
    id: int
    type: str
    message: str
    timestamp: datetime
    audio_data: Optional[str] = None
    audio_format: Optional[str] = "base64"
    mood_score: Optional[float] = None

class DashboardData(BaseModel):
    current_mood: float
    mood_trends: List[dict]
    conversation_counts: List[dict]
    conversations: List[Conversation]

# --------------- Helpers ----------------
def parse_logs():
    conversations = []
    files = sorted(os.listdir(LOGS_DIR))
    idx = 1
    for fname in files:
        path = os.path.join(LOGS_DIR, fname)
        if not fname.endswith(".csv"):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                timestamp_str, message = row
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
                # Try to attach WAV if exists
                wav_path = os.path.join(AUDIO_DIR, f"{idx}.wav")
                audio_data = None
                if os.path.exists(wav_path):
                    with open(wav_path, "rb") as wf:
                        audio_data = base64.b64encode(wf.read()).decode("utf-8")
                
                conversations.append(
                    Conversation(
                        id=idx,
                        type="bot",
                        message=message,
                        timestamp=timestamp,
                        audio_data=audio_data
                    )
                )
                idx += 1
    return conversations

# --------------- API ----------------
@app.get("/generate", response_model=DashboardData)
async def get_dashboard_data():
    conversations = parse_logs()
    current_mood = 7.2  # optional: calculate dynamically
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
