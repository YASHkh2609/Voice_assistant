import asyncio
from language_model import LanguageModelProcessor
from text_to_speech import TextToSpeech
from live_transcription import get_transcript
import signal
import sys

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.text_to_speech = TextToSpeech()
        self.stop_event = asyncio.Event()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
            print(f"User: {full_sentence}")  # Display user input

        # Handle interrupt signal
        def signal_handler(sig, frame):
            print("Interrupt received, stopping...")
            self.stop_event.set()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while not self.stop_event.is_set():
            await get_transcript(handle_full_sentence)
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response, elapsed_time = self.llm.process(self.transcription_response)
            print(f"JK: {llm_response} (Response time: {elapsed_time}ms)")  # Display AI response
            self.text_to_speech.speak(llm_response)
            self.transcription_response = ""
