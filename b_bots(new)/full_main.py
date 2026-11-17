import os
import time
import json
import csv
import threading
import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
import pyaudio
import requests
import re
import subprocess
from vosk import Model, KaldiRecognizer
from datetime import datetime, timedelta
from TTS.api import TTS

# ================== CONFIG ==================
MODEL_PATH = "models/keyword_model.tflite"
VOSK_MODEL = "models/vosk-model-en-us-0.22"
SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 13
MAX_LEN = 32
THRESHOLD = 0.90
SILENCE_TIMEOUT = 3
LOG_DIR = "logs"
LOG_RETENTION_DAYS = 7
MAX_REPEATS = 10
ASR_RUNNING = False
API_URL = "https://wishayut-uvicorn.pkongkha.org/generate"
ENABLE_LOGS = True
asr_lock = threading.Lock()

# ================== TTS CONFIG ==================
OUTPUT_DIR = "output_realtime"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# tts_model = TTS(model_name=MODEL_NAME)
# MODEL_NAME = "tts_models/en/ljspeech/overflow"
# tts_counter = 1
# tts_lock = threading.Lock()  # Prevent TTS overlapping

# ================== LOGGING ==================
os.makedirs(LOG_DIR, exist_ok=True)
current_user_log = None
current_ai_log = None
current_date_str = None

def get_daily_log_files():
    global current_user_log, current_ai_log, current_date_str
    today_str = datetime.now().strftime("%Y%m%d")
    if current_date_str != today_str or current_user_log is None:
        current_date_str = today_str
        current_user_log = os.path.join(LOG_DIR, f"user_log_{today_str}.csv")
        if not os.path.exists(current_user_log):
            with open(current_user_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_transcript"])

        current_ai_log = os.path.join(LOG_DIR, f"ai_log_{today_str}.csv")
        if not os.path.exists(current_ai_log):
            with open(current_ai_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "ai_reply"])
    return current_user_log, current_ai_log

def log_user_transcript(transcript):
    if not ENABLE_LOGS:
        return
    log_file, _ = get_daily_log_files()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, transcript])

def log_ai_reply(reply):
    if not ENABLE_LOGS:
        return
    _, log_file = get_daily_log_files()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, reply])

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

# ================== TTS FUNCTIONS ==================
# def get_next_counter():
#     global tts_counter
#     return tts_counter

# def speak_reply(text):
#     """Generate TTS audio and play it without overlapping."""
#     global tts_counter
#     with tts_lock:
#         output_file = os.path.join(OUTPUT_DIR, f"voice_{tts_counter}.wav")
#         tts_model.tts_to_file(text=text, file_path=output_file)
#         tts_counter += 1
#         if os.name == "nt":
#             audio_player = f"powershell -c (New-Object Media.SoundPlayer '{output_file}').PlaySync()"
#             subprocess.Popen(audio_player, shell=True)
#         else:
#             if subprocess.call("which ffplay", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
#                 subprocess.Popen(f"ffplay -nodisp -autoexit {output_file} > /dev/null 2>&1 &", shell=True)
#             else:
#                 subprocess.call(f"aplay {output_file}", shell=True)

# ================== API REQUEST ==================
def send_to_Blance_core(prompt_text):
    """
    Sends user transcript to server,
    receives AI response + audio, and plays it.
    """
    try:
        response = requests.post(API_URL, json={"prompt": prompt_text}, timeout=30, stream=True)
        if response.status_code == 200:
            # Get AI text from custom header
            reply = response.headers.get("X-AI-Response", "")
            log_ai_reply(reply)
            print(f"\n🤖 Gemini3 Reply: {reply}\n")

            # Save and play WAV
            audio_file = os.path.join(OUTPUT_DIR, f"server_audio_{int(time.time()*1000)}.wav")
            with open(audio_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            # Play audio
            if os.name == "nt":
                subprocess.Popen(f"powershell -c (New-Object Media.SoundPlayer '{audio_file}').PlaySync()", shell=True)
            else:
                if subprocess.call("which ffplay", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                    subprocess.Popen(f"ffplay -nodisp -autoexit {audio_file} > /dev/null 2>&1 &", shell=True)
                else:
                    subprocess.call(f"aplay {audio_file}", shell=True)

            return reply
        else:
            print(f"⚠️ API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to connect to Gemini3 API: {e}")
        return None


# ================== MODEL LOADING ==================
try:
    interpreter = tf.compat.v1.lite.Interpreter(model_path=MODEL_PATH)
except:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Keyword model loaded")

try:
    asr_model = Model(VOSK_MODEL)
    VOSK_AVAILABLE = True
    print("✅ Vosk ASR model loaded")
except Exception as e:
    print(f"⚠️ Vosk failed to load: {e}")
    VOSK_AVAILABLE = False

# ================== MFCC PREPROCESS ==================
def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

# ================== ASR FUNCTION ==================
def start_asr():
    global ASR_RUNNING
    with asr_lock:
        if ASR_RUNNING:
            print("ℹ️ ASR already running. Skipping new session.")
            return
        ASR_RUNNING = True

    if not VOSK_AVAILABLE:
        print("❌ Vosk not available")
        with asr_lock:
            ASR_RUNNING = False
        return

    print(f"🎙️ Speak now... (auto-stop after {SILENCE_TIMEOUT}s of silence)")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=4000)
    stream.start_stream()
    recognizer_local = KaldiRecognizer(asr_model, SAMPLE_RATE)
    transcript = ""
    last_speech_time = time.time()
    last_partial = ""
    repeat_count = 0

    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            has_speech = False

            if recognizer_local.AcceptWaveform(data):
                result = json.loads(recognizer_local.Result())
                text = result.get("text", "").strip()
                if text:
                    transcript += " " + text
                    has_speech = True
                    print(f"📝 {text}")
            else:
                partial = json.loads(recognizer_local.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text:
                    has_speech = True
                    if partial_text == last_partial:
                        repeat_count += 1
                        if repeat_count >= MAX_REPEATS:
                            print(f"⚠️ Repeated partial '{partial_text}' detected. Stopping ASR.")
                            break
                    else:
                        last_partial = partial_text
                        repeat_count = 0

            if has_speech:
                last_speech_time = time.time()
            if time.time() - last_speech_time > SILENCE_TIMEOUT:
                print(f"⏱️ No speech detected for {SILENCE_TIMEOUT}s. Stopping ASR.")
                break

    except KeyboardInterrupt:
        print("\n🛑 ASR stopped by user")

    final_result = json.loads(recognizer_local.FinalResult())
    final_text = final_result.get("text", "").strip()
    transcript += " " + final_text
    transcript = transcript.strip()

    if transcript:
        print(f"✅ Final Transcript: {transcript}")
        log_user_transcript(transcript)
        threading.Thread(target=lambda: send_to_Blance_core(transcript), daemon=True).start()
    else:
        print("ℹ️ No speech detected to log")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with asr_lock:
        ASR_RUNNING = False

# ================== KEYWORD DETECTION ==================
def detect_keyword():
    print("🎤 Listening for keyword... Press Ctrl+C to stop")
    while True:
        with asr_lock:
            if ASR_RUNNING:
                time.sleep(0.2)
                continue

        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = audio.flatten()
        mfcc_input = extract_mfcc(audio)
        interpreter.set_tensor(input_details[0]['index'], mfcc_input)
        interpreter.invoke()
        prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        prob_flipped = 1 - prob

        if prob_flipped > THRESHOLD:
            print(f"✅ Keyword detected! (prob={prob_flipped:.3f})")
            start_asr()
        else:
            print(f"❌ Not detected (prob={prob_flipped:.3f})")

        time.sleep(0.1)

# ================== DAILY CLEANUP THREAD ==================
threading.Thread(target=lambda: cleanup_old_logs(LOG_RETENTION_DAYS), daemon=True).start()

# ================== START ==================
if __name__ == "__main__":
    try:
        detect_keyword()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
