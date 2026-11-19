import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
from config import MODEL_PATH, SAMPLE_RATE, DURATION, N_MFCC, MAX_LEN, THRESHOLD

# ================== KEYWORD DETECTION CLASS ==================
class KeywordDetector:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()

    def load_model(self):
        """Load the TensorFlow Lite model."""
        try:
            self.interpreter = tf.compat.v1.lite.Interpreter(model_path=MODEL_PATH)
        except:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("✅ Keyword model loaded")

    def extract_mfcc(self, audio):
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

    def detect_keyword_once(self):
        """Detect keyword in a single audio sample."""
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = audio.flatten()
        mfcc_input = self.extract_mfcc(audio)

        self.interpreter.set_tensor(self.input_details[0]['index'], mfcc_input)
        self.interpreter.invoke()
        prob = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        prob_flipped = 1 - prob

        return prob_flipped > THRESHOLD, prob_flipped

    def detect_keyword_loop(self, asr_callback, asr_lock):
        """Continuously listen for keyword detection."""
        print("🎤 Listening for keyword... Press Ctrl+C to stop")
        while True:
            with asr_lock:
                # Check if ASR is already running
                pass
            
            detected, prob = self.detect_keyword_once()
            
            if detected:
                print(f"✅ Keyword detected! (prob={prob:.3f})")
                asr_callback()
            else:
                print(f"❌ Not detected (prob={prob:.3f})")

            time.sleep(0.1)