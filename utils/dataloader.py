import os
from pathlib import Path

class DataLoader:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)

    def list_videos(self):
        video_files = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            video_files.extend(self.base_dir.rglob(ext))
        return [str(v) for v in video_files]
