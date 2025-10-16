import torch
import torch.nn as nn
import torchaudio


class CNNRNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B,16,H/2,W/2]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B,32,H/4,W/4]
        )
        # We don’t know the CNN output size until runtime → use lazy
        self.rnn = nn.LSTM(input_size=32 * 16, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B,1,H,W] -> CNN -> [B,32,H/4,W/4]
        feat = self.cnn(x)
        B, C, H, W = feat.shape
        # treat time dimension as W, flatten C*H into features
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B,W,C,H]
        feat = feat.view(B, W, C * H)  # [B, seq_len, feat_dim]
        out, _ = self.rnn(feat)  # [B, seq_len, hidden]
        out = out[:, -1, :]  # last timestep
        return self.fc(out)


class AdvancedAudioDeepfakeProcessor:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.model = CNNRNNModel(num_classes=2).to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def detect_deepfake_audio(self, audio_path: str):
        waveform, sr = torchaudio.load(audio_path)

        # Resample for consistency
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        # Convert to mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=400, hop_length=160, n_mels=64
        )(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel).squeeze(0)  # [H,W]

        # Split into overlapping chunks (~1s windows)
        hop = 64
        chunks = []
        for i in range(0, mel_db.shape[1] - hop + 1, hop):
            chunk = mel_db[:, i:i + hop]  # [64,64]
            chunks.append(chunk.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]

        # Fallback for very short audio
        if not chunks:
            chunks.append(mel_db.unsqueeze(0).unsqueeze(0))

        # Run model
        probs = []
        with torch.no_grad():
            for c in chunks:
                out = self.model(c.to(self.device))
                prob = torch.softmax(out, dim=-1).cpu().numpy()[0]
                probs.append(prob)

        mean_prob = sum(probs) / len(probs)
        return {
            "Real Probability": float(mean_prob[0]),
            "Fake Probability": float(mean_prob[1]),
            "Prediction": "FAKE" if mean_prob[1] > mean_prob[0] else "REAL"
        }
