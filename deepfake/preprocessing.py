import cv2
import numpy as np

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_idxs:
            frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames, target_size=(112, 112)):
    processed = []
    for f in frames:
        f = cv2.resize(f, target_size)
        f = f / 255.0  # normalize to [0,1]
        processed.append(f)
    return np.array(processed, dtype=np.float32)  # (T, H, W, C)
