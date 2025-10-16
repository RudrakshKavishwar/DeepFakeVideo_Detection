import torch
from transformers import pipeline

class AudioTranscriber:
    def __init__(self, device="cpu"):
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1
        self.pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=self.device)

    def transcribe(self, audio_path: str):
        """
        Transcribes audio into text.
        """
        print(f"[DEBUG] Transcribing: {audio_path}")
        result = self.pipe(audio_path)
        return result["text"]
