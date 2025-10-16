from moviepy.editor import VideoFileClip

def add_audio_back(video_path: str, output_path: str):
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        print("⚠️ No audio track found in the input video!")
    else:
        # Reattach audio explicitly
        final = clip.set_audio(clip.audio)
        final.write_videofile(
            output_path,
            codec="libx264",      # widely supported
            audio_codec="aac",   # ensure audio is written
            temp_audiofile="temp-audio.m4a", 
            remove_temp=True
        )
