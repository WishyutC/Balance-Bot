import os
import time
import subprocess
import base64
import requests
from config import API_URL, OUTPUT_DIR
from logger import log_ai_reply

# ================== API REQUEST ==================
def send_to_Blance_core(prompt_text):
    """
    Sends user transcript to server,
    receives AI response + audio (base64), decodes and plays it.
    """
    try:
        response = requests.post(API_URL, json={"prompt": prompt_text}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            reply = data.get("response_text", "")
            audio_b64 = data.get("audio_base64", "")

            # Log and display text reply
            log_ai_reply(reply)
            print(f"\n🤖 Gemini3 Reply: {reply}\n")

            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)
            audio_file = os.path.join(OUTPUT_DIR, f"server_audio_{int(time.time()*1000)}.wav")
            with open(audio_file, "wb") as f:
                f.write(audio_bytes)

            # Play audio
            play_audio_file(audio_file)
            return reply
        else:
            print(f"⚠️ API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to connect to Gemini3 API: {e}")
        return None

def play_audio_file(audio_file):
    """Play audio file using system-appropriate player."""
    if os.name == "nt":
        subprocess.Popen(f"powershell -c (New-Object Media.SoundPlayer '{audio_file}').PlaySync()", shell=True)
    else:
        if subprocess.call("which ffplay", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            subprocess.Popen(f"ffplay -nodisp -autoexit '{audio_file}' > /dev/null 2>&1 &", shell=True)
        else:
            subprocess.call(f"aplay '{audio_file}'", shell=True)

def initialize_api_client():
    """Initialize API client directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
