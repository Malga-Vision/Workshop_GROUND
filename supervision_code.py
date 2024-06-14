import supervision as sv
import numpy as np
from ultralytics import YOLO



model = YOLO("yolov8x.pt")


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = results = model.predict("/home/federico/Pictures/vlcsnap-2024-06-12-17h53m35s052.png", conf=0.25, device="cuda")[0]
    with open('')
    return results

VIDEO_PATH = "/home/federico/Videos/video.mp4"
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
sv.process_video(source_path=VIDEO_PATH, target_path=f"/home/federico/Videos/result.mp4", callback=process_frame)