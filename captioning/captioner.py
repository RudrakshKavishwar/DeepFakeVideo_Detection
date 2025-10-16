import torch
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class AdvancedVideoCaptioner:
    def __init__(self, device=None):
        # Auto-detect GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[🚀 Captioner using device: {self.device}]")

        # Load BLIP base model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.model.eval()

    def generate_detailed_captions(self, video_path, frame_step=20, max_frames=200):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 0
        captions = []

        print(f"[🎞️ Processing video frames: {total_frames}, FPS={fps}]")

        while True:
            ret, frame = cap.read()
            if not ret or frame_id > max_frames * frame_step:
                break

            if frame_id % frame_step == 0:
                # Convert frame to RGB Image
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=30,
                        num_beams=3,
                        temperature=0.7
                    )
                caption = self.processor.decode(out[0], skip_special_tokens=True)

                captions.append({
                    "timestamp": frame_id / fps if fps > 0 else frame_id,
                    "caption": caption
                })

            frame_id += 1

        cap.release()
        print(f"[📝 Generated {len(captions)} captions]")
        return {"captions": captions, "actions": []}
