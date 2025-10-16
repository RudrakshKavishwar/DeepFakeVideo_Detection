import sys
from pathlib import Path
import argparse
import os
import webbrowser

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from deepfake.processor import AdvancedDeepfakeProcessor   # CNN–RNN for video
from captioning.captioner import AdvancedVideoCaptioner
from knowledge_graph.pipeline import build_and_visualize_kg
from utils.dataloader import DataLoader
from audio.processor import AdvancedAudioDeepfakeProcessor  # CNN–RNN for audio

# For audio transcription
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 🎵 For video + audio export
from moviepy.editor import VideoFileClip


def add_audio_back(input_video_path: str, output_video_path: str):
    """Ensure the processed video keeps original audio."""
    clip = VideoFileClip(input_video_path)

    if clip.audio is None:
        print("⚠️ No audio track found in the input video!")
        clip.write_videofile(output_video_path, codec="libx264")
    else:
        # Explicitly reattach audio track
        final = clip.set_audio(clip.audio)
        final.write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",             # ensure audio track is kept
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )


# ------------------- VIDEO PIPELINE -------------------
def run_pipeline(video_path: str):
    print("=" * 60)
    print(f"🎬 Processing video: {video_path}")
    print("=" * 60)

    # Step 1: Deepfake Detection (CNN–RNN hybrid)
    df_proc = AdvancedDeepfakeProcessor(device="cpu")
    df_result = df_proc.detect_deepfake_advanced(video_path)
    print("\n[🔍 Deepfake Detection Result]")
    print(df_result)

    # Step 2: Captioning (real captions extracted)
    cap_proc = AdvancedVideoCaptioner(device="cpu")
    cap_result = cap_proc.generate_detailed_captions(video_path, frame_step=30)
    print("\n[📝 Captions]")
    for c in cap_result["captions"]:
        ts = c.get("timestamp", 0.0)
        txt = c.get("caption", "").strip()
        print(f" - {ts:.2f}s: {txt}")

    # Step 3: Knowledge Graph
    segments = [
        {
            "timestamp": c.get("timestamp", 0.0),
            "caption": c.get("caption", "").strip(),
            "actions": []
        }
        for c in cap_result["captions"]
    ]

    stats, msg = build_and_visualize_kg(segments, "data/graph.html")
    print("\n[🧠 Knowledge Graph]")
    print(stats)
    print(f"Graph visualization saved to {msg}")

    # 🎵 Save final video with audio retained
    output_path = str(Path("data/processed") / (Path(video_path).stem + "_with_audio.mp4"))
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    add_audio_back(video_path, output_path)
    print(f"\n🎧 Final video with audio saved at: {output_path}")

    # ▶️ Auto-play video using default media player
    try:
        os.startfile(output_path)  # Windows
    except AttributeError:
        os.system(f"open '{output_path}'")  # macOS/Linux fallback

    # 🌐 Auto-open Knowledge Graph in browser
    graph_path = str(Path("data/graph.html").resolve())
    print(f"\n🌐 Opening Knowledge Graph visualization...")
    webbrowser.open(f"file:///{graph_path}")


# ------------------- AUDIO PIPELINE -------------------
def run_pipeline_audio(audio_path: str):
    print("=" * 60)
    print(f"🎤 Processing audio: {audio_path}")
    print("=" * 60)

    # Step 1: Deepfake Detection (CNN–RNN hybrid)
    audio_proc = AdvancedAudioDeepfakeProcessor(device="cpu")
    df_result = audio_proc.detect_deepfake_audio(audio_path)
    print("\n[🔍 Deepfake Detection Result]")
    print(df_result)

    # Step 2: Transcription (real captions from audio)
    print("\n[📝 Transcription]")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(pred_ids[0])
    print(f" - {transcription.strip()}")


# ------------------- MAIN -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--audio", type=str, help="Path to a single audio file")
    parser.add_argument("--all", action="store_true", help="Process all videos in data/samples/")
    args = parser.parse_args()

    data = DataLoader("data")
    videos = data.list_videos()

    if args.video:             # 🎥 Only video pipeline
        run_pipeline(args.video)
    elif args.audio:           # 🎤 Only audio pipeline
        run_pipeline_audio(args.audio)
    elif args.all:             # 🎥 Process all videos
        if not videos:
            print("⚠️ No videos found in data/. Please add videos to data/samples/")
        else:
            for v in videos:
                run_pipeline(v)
    else:                      # Default: process first video if nothing specified
        if not videos:
            print("⚠️ No videos found in data/. Please add videos to data/samples/")
        else:
            run_pipeline(videos[0])
