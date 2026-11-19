import time
from typing import Any

import ai_edge_litert.interpreter as litert
import librosa
import librosa.feature
import numpy as np
import pyaudio

from config import KEYWORD_DETECTION_SPLIT, MODEL_PATH, SAMPLE_RATE, DURATION, N_MFCC, MAX_LEN, THRESHOLD


# ================== KEYWORD DETECTION CLASS ==================
class KeywordDetector:
    def __init__(self):
        self.interpreter = None
        self.input_details: list[Any] = []
        self.output_details: list[Any] = []
        self.load_model()

    def load_model(self):
        self.interpreter = litert.Interpreter(model_path=MODEL_PATH)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("✅ Keyword model loaded")

    @staticmethod
    def extract_mfcc(audio):
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

    @staticmethod
    def preprocess_audio(audio, target_length):  # Changed from SAMPLE_RATE * DURATION
        """Preprocess audio for keyword detection."""
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        audio_int16 = (audio * 32767).astype(np.int16)
        audio = audio_int16.astype(np.float32) / 32768.0
        return audio

    def detect_keyword_once(self, stream: pyaudio.Stream):
        """Detect keyword in a single audio sample."""
        chunk_size = int(SAMPLE_RATE * DURATION / KEYWORD_DETECTION_SPLIT)
        raw_audio = stream.read(chunk_size, exception_on_overflow=False)
        audio = np.frombuffer(raw_audio, dtype=np.float32)
        audio = self.preprocess_audio(audio, chunk_size)
        mfcc_input = self.extract_mfcc(audio)

        self.interpreter.set_tensor(self.input_details[0]['index'], mfcc_input)
        self.interpreter.invoke()
        prob = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        prob_flipped = 1 - prob

        return prob_flipped > THRESHOLD, prob_flipped

    def detect_keyword_loop(self, stream, asr_callback, asr_lock):
        """Continuously listen for keyword detection."""
        print("🎤 Listening for keyword... Press Ctrl+C to stop")
        while True:
            with asr_lock:
                # Check if ASR is already running
                pass

            if not stream.is_active():
                stream.start_stream()
            detected, prob = self.detect_keyword_once(stream)

            if detected:
                stream.stop_stream()
                print(f"✅ Keyword detected! (prob={prob:.3f})")
                asr_callback(stream)
            else:
                print(f"❌ Not detected (prob={prob:.3f})")
