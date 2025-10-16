import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)

class SimpleTransformerCaptioner(nn.Module):
    """Tiny captioning model for demo"""
    def __init__(self, vocab_size=30522, hidden_dim=256):
        super().__init__()
        self.encoder = VideoEncoder()
        self.decoder = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, video_features, tgt):
        enc_out = self.encoder(video_features)
        dec_out = self.decoder(tgt, enc_out.unsqueeze(0))
        return self.fc(dec_out)
