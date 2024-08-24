import os
import requests
import subprocess
import shutil
from dotenv import load_dotenv
import threading
import time

load_dotenv()

class TextToSpeech:
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-asteria-en"

    def __init__(self):
        self.player_process = None
        self.stop_event = threading.Event()

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        return shutil.which(lib_name) is not None

    def _play_audio(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"text": text}

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        self.player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    self.player_process.stdin.write(chunk)
                    self.player_process.stdin.flush()
                    if self.stop_event.is_set():
                        break

        if self.player_process.stdin:
            self.player_process.stdin.close()
        self.player_process.wait()

    def speak(self, text):
        # Stop any ongoing speech before starting new one
        self.stop()
        self.stop_event.clear()
        
        # Start a new thread for playing audio
        self.speech_thread = threading.Thread(target=self._play_audio, args=(text,))
        self.speech_thread.start()

    def stop(self):
        # Signal the audio playback to stop
        self.stop_event.set()
        if self.player_process:
            self.player_process.terminate()
            self.player_process.wait()

# Example usage:
# if __name__ == "__main__":
#     tts = TextToSpeech()
#     tts.speak("Hello, how can I assist you today?")
#     time.sleep(10)  # Simulate some delay before stopping
#     tts.stop()
