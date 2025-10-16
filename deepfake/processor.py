import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2


class CNNRNNVideoModel(nn.Module):
    def __init__(self, cnn_out_dim=512, hidden_dim=256, num_classes=2):
        super().__init__()
        # Simple CNN (replace with ResNet for stronger baseline)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_cnn = nn.Linear(64, cnn_out_dim)

        # RNN on sequence of frame features
        self.rnn = nn.LSTM(cnn_out_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        feats = self.cnn(frames).view(B * T, -1)
        feats = self.fc_cnn(feats)

        feats = feats.view(B, T, -1)
        rnn_out, _ = self.rnn(feats)
        logits = self.fc_out(rnn_out[:, -1, :])
        return logits


class AdvancedDeepfakeProcessor:
    def __init__(self, device="cpu", model_path=None):
        self.device = device
        self.model = CNNRNNVideoModel().to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])

    def detect_deepfake_advanced(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)
            if len(frames) >= 16:  # limit sequence length
                break
        cap.release()

        if not frames:
            return {"error": "No frames extracted"}

        frames = torch.stack(frames).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        with torch.no_grad():
            logits = self.model(frames)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return {
            "Real Probability": float(probs[0]),
            "Fake Probability": float(probs[1]),
            "Prediction": "FAKE" if probs[1] > probs[0] else "REAL"
        }
