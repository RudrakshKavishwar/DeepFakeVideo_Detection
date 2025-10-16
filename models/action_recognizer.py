import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np

class ActionRecognizer:
    def __init__(self, device="cpu"):
        self.device = device
        # Load pretrained ResNet3D model
        self.model = torchvision.models.video.r3d_18(weights="KINETICS400_V1").to(device)
        self.model.eval()

        # Define preprocessing (resize → tensor → normalize)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])

        # Load Kinetics-400 class names
        kinetics_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/label_map.txt"
        try:
            import requests
            labels = requests.get(kinetics_url).text.strip().split("\n")
            self.classes = [l.split(":")[-1].strip().strip('"') for l in labels]
        except Exception:
            self.classes = [f"class_{i}" for i in range(400)]

    def recognize_actions(self, frames, sample_size=16):
        """
        frames: list of numpy arrays (BGR)
        sample_size: number of frames to use for recognition
        """
        if len(frames) < sample_size:
            return []

        # Sample evenly spaced frames
        idxs = np.linspace(0, len(frames)-1, sample_size).astype(int)
        sampled_frames = [frames[i] for i in idxs]

        # Apply transforms
        processed = [self.transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in sampled_frames]
        clip = torch.stack(processed)  # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1, C, T, H, W)

        with torch.no_grad():
            preds = self.model(clip)
            probs = torch.nn.functional.softmax(preds, dim=1)[0]

        # Top-3 predictions
        topk = torch.topk(probs, k=3)
        results = []
        for score, idx in zip(topk.values, topk.indices):
            results.append({
                "action": self.classes[idx] if idx < len(self.classes) else f"class_{idx}",
                "confidence": float(score),
                "start_time": 0.0,
                "end_time": float(len(frames) / 30)  # rough duration
            })

        return results
