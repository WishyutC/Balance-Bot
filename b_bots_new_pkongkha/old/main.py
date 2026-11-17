import threading
from keyword_detector import KeywordDetector
from speech_recognizer import SpeechRecognizer
from logger import initialize_logging, cleanup_old_logs
from api_client import initialize_api_client
from tts_module import TTSModule
from config import LOG_RETENTION_DAYS

# ================== MAIN APPLICATION ==================
class VoiceAssistant:
    def __init__(self):
        self.keyword_detector = KeywordDetector()
        self.speech_recognizer = SpeechRecognizer()
        self.tts_module = TTSModule()
        self.initialize_components()

    def initialize_components(self):
        """Initialize all components."""
        initialize_logging()
        initialize_api_client()
        
        # Start cleanup thread
        threading.Thread(
            target=lambda: cleanup_old_logs(LOG_RETENTION_DAYS), 
            daemon=True
        ).start()

    def start_listening(self):
        """Start the main listening loop."""
        try:
            self.keyword_detector.detect_keyword_loop(
                asr_callback=self.speech_recognizer.start_asr,
                asr_lock=self.speech_recognizer.asr_lock
            )
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")

def main():
    """Main entry point."""
    assistant = VoiceAssistant()
    assistant.start_listening()

if __name__ == "__main__":
    main()