import os
import subprocess
import threading
from config import OUTPUT_DIR

# ================== TTS MODULE ==================
class TTSModule:
    def __init__(self):
        # Commented out as per original code
        # self.tts_model = TTS(model_name=MODEL_NAME)
        # self.MODEL_NAME = "tts_models/en/ljspeech/overflow"
        self.tts_counter = 1
        self.tts_lock = threading.Lock()

    def get_next_counter(self):
        """Get next TTS counter value."""
        return self.tts_counter

    def speak_reply(self, text):
        """Generate TTS audio and play it without overlapping."""
        # This function is commented out in the original code
        # Keeping it here for reference but commented out
        
        # with self.tts_lock:
        #     output_file = os.path.join(OUTPUT_DIR, f"voice_{self.tts_counter}.wav")
        #     self.tts_model.tts_to_file(text=text, file_path=output_file)
        #     self.tts_counter += 1
        #     if os.name == "nt":
        #         audio_player = f"powershell -c (New-Object Media.SoundPlayer '{output_file}').PlaySync()"
        #         subprocess.Popen(audio_player, shell=True)
        #     else:
        #         if subprocess.call("which ffplay", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        #             subprocess.Popen(f"ffplay -nodisp -autoexit {output_file} > /dev/null 2>&1 &", shell=True)
        #         else:
        #             subprocess.call(f"aplay {output_file}", shell=True)
        pass