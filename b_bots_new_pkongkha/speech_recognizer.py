import pstats
import time
import json
import threading

import numpy as np
import pyaudio

from mood_recognizer import Model as MoodModel
from numpy.typing import NDArray
from vosk import Model, KaldiRecognizer
from config import DURATION, VOSK_MODEL, SAMPLE_RATE, SILENCE_TIMEOUT, MAX_REPEATS, MOOD_MODEL
from emotion import Emotion
from logger import log_user_transcript
from api_client import send_to_balance_core

# ================== SPEECH RECOGNIZER CLASS ==================
class SpeechRecognizer:
    def __init__(self):
        self.asr_model = None
        self.vosk_available = False
        self.asr_running = False
        self.asr_lock = threading.Lock()
        self.mood_recognizer = None
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

        try:
            self.mood_recognizer = MoodModel()
            self.mood_recognizer.load_model(MOOD_MODEL)
            print("✅ Mood recognition model loaded")
        except Exception as e:
            print(f"⚠️ Mood recognition model failed to load: {e}")
            self.mood_recognizer = None

    def is_running(self):
        """Check if ASR is currently running."""
        with self.asr_lock:
            return self.asr_running

    def start_asr(self, stream: pyaudio.Stream):
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

        audio_chunk: NDArray = np.array([], dtype=np.float32)
        emotion: Emotion = Emotion(Emotion.NEUTRAL)
        emotion_lock = threading.Lock()
        emotion_history: list[int] = []
        stop_mood_event = threading.Event()

        def recognize_mood_worker(stop_event: threading.Event):
            nonlocal emotion, emotion_lock
            if not self.mood_recognizer:
                return
            while not stop_event.is_set():
                try:
                    features = self.mood_recognizer.extract_2d_features(audio_chunk, SAMPLE_RATE)
                    prediction = self.mood_recognizer.predict(features, verbose=0)
                    emotion = Emotion(np.argmax(prediction))
                    emotion_history.append(emotion.value)
                    print(f"😊 Detected Mood: {emotion.name}")
                except Exception as e:
                    print(f"⚠️ Mood recognition error: {e}")
                time.sleep(DURATION)
            emotion = Emotion(Emotion.NEUTRAL)

        if self.mood_recognizer:
            mood_thread = threading.Thread(target=recognize_mood_worker, args=(stop_mood_event,), daemon=True)
            mood_thread.start()

        recognizer_local = KaldiRecognizer(self.asr_model, SAMPLE_RATE)
        transcript = ""
        last_speech_time = time.time()
        last_partial = ""
        repeat_count = 0
        stream.start_stream()
        stream.read(int(DURATION * SAMPLE_RATE))
        print(f"🎙️ Speak now... (auto-stop after {SILENCE_TIMEOUT}s of silence)")

        try:
            while True:
                data = stream.read(int(DURATION * SAMPLE_RATE))
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                has_speech = False
                if recognizer_local.AcceptWaveform((audio_chunk * 32767).astype(np.int16).tobytes()):
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

        finally:
            stream.stop_stream()

        # Get final result
        final_result = json.loads(recognizer_local.FinalResult())
        final_text = final_result.get("text", "").strip()
        transcript += " " + final_text
        transcript = transcript.strip()

        if transcript:
            print(f"✅ Final Transcript: {transcript}")
            stop_mood_event.set()
            emotion_str = Emotion(max(emotion_history, key=emotion_history.count)).name if emotion_history else ""
            with emotion_lock:
                print(f"😊 Final Mood: {emotion_str}")
            log_user_transcript(transcript, emotion_str)
            send_to_balance_core(transcript)
        else:
            print("ℹ️ No speech detected to log")

        stop_mood_event.set()
        emotion_history.clear()

        with self.asr_lock:
            self.asr_running = False
