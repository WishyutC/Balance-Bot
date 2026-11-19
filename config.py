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
API_URL = "https://wishayut-uvicorn.pkongkha.org/generate"
ENABLE_LOGS = True

# ================== TTS CONFIG ==================
OUTPUT_DIR = "audio_output"

