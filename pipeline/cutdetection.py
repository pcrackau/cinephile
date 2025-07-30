import os
import cv2
import subprocess
from pathlib import Path

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


class CutDetector:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def detect_scenes(self, video_path: str):
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()
        video_manager.release()

        return [(int(start.get_frames()), int(end.get_frames())) for start, end in scene_list]

    # TODO return start/end-time of cut for json store
    def extract_cut(self, input_path: Path, start_frame: int, end_frame: int, fps: float, output_path: Path):
        start_time = self._frames_to_timecode(start_frame, fps)
        duration = (end_frame - start_frame) / fps

        subprocess.run([
            "ffmpeg",
            "-i", str(input_path),
            "-ss", start_time,
            "-t", f"{duration:.3f}",
            str(output_path)
        ], check=True)


    def save_all_cuts(self, video_path: Path, cuts: list[tuple[int, int]], output_dir: Path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        output_dir.mkdir(parents=True, exist_ok=True)

        for i, (start, end) in enumerate(cuts):
            output_path = output_dir / f"{video_path.stem}_cut_{i+1:03}.mp4"
            
            if output_path.exists():
                os.remove(output_path)
            self.extract_cut(video_path, start, end, fps, output_path)
            

    def _frames_to_timecode(self, frame_num: int, fps: float) -> str:
        seconds = frame_num / fps
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hrs:02}:{mins:02}:{secs:06.3f}"