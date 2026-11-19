import base64
import os
import wave
from datetime import datetime

import pyaudio
import requests
from pydub import AudioSegment  # type: ignore

from config import API_URL, OUTPUT_DIR
from logger import log_ai_reply

# New directory for browser-ready PCM 16-bit WAVs
AUDIO_PCM_DIR = "audio_pcm"
os.makedirs(AUDIO_PCM_DIR, exist_ok=True)


# ================== HELPER: Convert WAV to 16-bit PCM ==================
def convert_to_pcm16bit(raw_file: str) -> str:
    """
    Converts a WAV file to 16-bit PCM, mono, 44.1kHz, saves in OUTPUT_PCM_DIR.
    Returns the path to the PCM file.
    """
    timestamp_str = os.path.basename(raw_file).replace("B_bot_reply", "").replace(".wav", "")
    pcm_file = os.path.join(AUDIO_PCM_DIR, f"B_bot_reply{timestamp_str}.wav")

    # Load raw WAV
    sound = AudioSegment.from_file(raw_file)
    # Convert to 16-bit PCM
    sound = sound.set_frame_rate(44100).set_sample_width(2).set_channels(1)
    # Export to PCM directory
    sound.export(pcm_file, format="wav")
    return pcm_file


# ================== API REQUEST ==================
def send_to_balance_core(prompt_text):
    """
    Sends user transcript to server,
    receives AI response + audio (base64), decodes and plays it.
    """
    try:
        response = requests.post(API_URL, json={"prompt": prompt_text}, timeout=60)
        if response.status_code == 200:
            print(response)
            data = response.json()
            reply = data.get("response_text", "")
            audio_b64 = data.get("audio_base64", "")

            # Log and display text reply
            log_ai_reply(reply)
            print(f"\n🤖 Balance bot Reply: {reply}\n")

            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)

            # Create timestamped filename
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            raw_file = os.path.join(OUTPUT_DIR, f"B_bot_reply{timestamp_str}.wav")

            # Save raw audio
            with open(raw_file, "wb") as f:
                f.write(audio_bytes)

            # Convert to browser-ready PCM 16-bit WAV
            pcm_file = convert_to_pcm16bit(raw_file)

            # Play PCM audio
            play_audio_file(pcm_file)

            return reply
        else:
            print(f"⚠️ API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to connect to Gemini3 API: {e}")
        return None


# ================== AUDIO PLAYBACK ==================
def play_audio_file(audio_file):
    """Play audio file using pyaudio."""
    chunk = 1024
    wf = wave.open(audio_file, 'rb')
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                     channels=wf.getnchannels(),
                     rate=wf.getframerate(),
                     output=True)
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf.close()


# ================== INITIALIZE ==================
def initialize_api_client():
    """Initialize API client directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_PCM_DIR, exist_ok=True)
