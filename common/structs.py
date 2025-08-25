from pathlib import Path
import json


class CutSegment:
    def __init__(self, cut_id: int, segment_path: Path, metadata: dict = None):
        self.cut_id = cut_id
        self.segment_path = segment_path
        self.metadata = metadata or {}
        self.keyframes = []

    def add_keyframe(self, keyframe_info: dict):
        """Store detected keyframe information (shot type, objects, caption)."""
        self.keyframes.append(keyframe_info)


class Film:
    def __init__(self, video_path: Path, metadata_path: Path):
        self.video_path = video_path
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()
        self.cut_segments: list[CutSegment] = []

    def _load_metadata(self) -> dict:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {}

    def add_cut_segment(self, cut_segment: CutSegment):
        self.cut_segments.append(cut_segment)

