import time
import json
import threading
import pyaudio
from vosk import Model, KaldiRecognizer
from config import VOSK_MODEL, SAMPLE_RATE, SILENCE_TIMEOUT, MAX_REPEATS
from logger import log_user_transcript
from api_client import send_to_Blance_core

# ================== SPEECH RECOGNIZER CLASS ==================
class SpeechRecognizer:
    def __init__(self):
        self.asr_model = None
        self.vosk_available = False
        self.asr_running = False
        self.asr_lock = threading.Lock()
        self.load_model()

    def load_model(self):
        """Load the Vosk ASR model."""
        try:
            self.asr_model = Model(VOSK_MODEL)
            self.vosk_available = True
            print("✅ Vosk ASR model loaded")
        except Exception as e:
            print(f"⚠️ Vosk failed to load: {e}")
            self.vosk_available = False

    def is_running(self):
        """Check if ASR is currently running."""
        with self.asr_lock:
            return self.asr_running

    def start_asr(self):
        """Start Automatic Speech Recognition."""
        with self.asr_lock:
            if self.asr_running:
                print("ℹ️ ASR already running. Skipping new session.")
                return
            self.asr_running = True

        if not self.vosk_available:
            print("❌ Vosk not available")
            with self.asr_lock:
                self.asr_running = False
            return

        print(f"🎙️ Speak now... (auto-stop after {SILENCE_TIMEOUT}s of silence)")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, 
                       input=True, frames_per_buffer=4000)
        stream.start_stream()
        recognizer_local = KaldiRecognizer(self.asr_model, SAMPLE_RATE)
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

        # Get final result
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

        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()

        with self.asr_lock:
            self.asr_running = False