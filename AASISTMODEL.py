# AASISTMODEL.py
import torch
import librosa
import numpy as np
import sounddevice as sd
from models import AASIST
import warnings
warnings.filterwarnings("ignore")


class LiveAASISTDetector:
    def __init__(self, model_path="models/weights/AASIST.pth"):
        self.device = torch.device("cpu")

        # Audio parameters
        self.sample_rate = 16000
        self.duration = 4.0  # seconds
        self.chunk_samples = int(self.sample_rate * self.duration)

        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()

    def load_model(self, model_path):
        """Load pretrained AASIST model"""

        d_args = {
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.5, 0.5],
            "temperatures": [1.0, 1.0, 1.0],
            "first_conv": 5
        }

        model = AASIST(d_args)

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    def preprocess_audio(self, audio):
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.flatten()
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        target_len = int(self.sample_rate * self.duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        return torch.tensor(audio, dtype=torch.float32)
    
    def detect(self, audio_chunk):
        with torch.no_grad():
            waveform = self.preprocess_audio(audio_chunk)
            waveform = waveform.unsqueeze(0).to(self.device)
            _, logits = self.model(waveform)
            prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
            return prob_fake

    def live_detection(self):
        """Run real-time detection from microphone"""
        print("\nStarting live AASIST detection")
        print(f"Window size: {self.duration}s | Sample rate: {self.sample_rate}Hz")
        print("Press Ctrl+C to stop\n")

        def audio_callback(indata, frames, time, status):
            if status:
                print(status)

            audio = indata.copy()
            prob = self.detect(audio)

            label = "FAKE" if prob > 0.006 else "REAL"
            print(f"\rFake prob: {prob:.3f} | {label}", end="")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_samples,
            callback=audio_callback
        ):
            while True:
                sd.sleep(1000)


if __name__ == "__main__":
    try:
        detector = LiveAASISTDetector()
        detector.live_detection()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
