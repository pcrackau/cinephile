import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import json
import uuid

from langchain_core.documents import Document



def flatten_for_rag(data, parent_key=""):
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{parent_key} ({k})" if parent_key else k
            lines.extend(flatten_for_rag(v, full_key))
    elif isinstance(data, list):
        for item in data:
            lines.extend(flatten_for_rag(item, parent_key))
    else:
        lines.append(f"{parent_key}: {data}")
    return lines

class CutSegment:
    def __init__(self, film_id: str, cut_file: Path, start_time: str, end_time: str, duration: float,
                 start_frame: int, end_frame: int, fps: float):
        self.film_id = film_id
        self.id = str(uuid.uuid4())
        self.cut_file = Path(cut_file)
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps
        self.keyframes = []
        self.metadata: dict = {}

    def to_dict(self):
        return {
            "cut_file": str(self.cut_file),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "fps": self.fps,
            "keyframes": self.keyframes,
            "metadata": self.metadata,
        }
    
    def generate_cut_doc(self) -> Document:
        objects = [obj.get("label") for obj in self.metadata.get("objects", [])]
        shot_type = self.metadata.get("shot_type")

        page_content = "\n".join(flatten_for_rag({
            "cutsegment": f"{self.start_time} – {self.end_time}",
            "objects": objects,
            "shot_type": shot_type,
        }))

        return Document(
            page_content=page_content,
            metadata={
                "cut_id": self.id,
                "film_id": self.film_id,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.duration,
            }
        )

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            cut_file=data["cut_file"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            duration=data["duration"],
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            fps=data["fps"],
            keyframes=data.get("keyframes", []),
            metadata=data.get("metadata", {})
        )
    
    def add_keyframe(self, frame_id: int, image_path: Path, timestamp: float,
                     objects: list[str] = None, shot_type: str = None):
        self.keyframes.append({
            "frame_id": frame_id,
            "image_path": str(image_path),
            "timestamp": timestamp,
            "objects": objects or [],
            "shot_type": shot_type
        })


class Film:
    def __init__(self, video_file: Path, metadata_file: Path):
        self.video_file = video_file
        with open(metadata_file) as f:
            raw_metadata = json.load(f)
        self.metadata = self._curate_metadata(raw_metadata)

        self.processing_dir = Path("processing") / self.video_file.stem
        self.cuts_output_dir = self.processing_dir / "cuts"
        self.cuts_json = self.processing_dir / "cuts.json"
        self.id = str(uuid.uuid4())
        self.cuts: list[CutSegment] = []
        self.transitions = []

    
    def _curate_metadata(self, raw):
        return {
            "title": raw.get("dcTitleLangAware"),
            "country": raw.get("country"),
            "year": raw.get("edmTimespan"),
            "production": raw.get("dcCreatorLangAware"),
            "blurb": raw.get("dcDescription"),
            "language": raw.get("dcLanguage"),
            "dataProvider": raw.get("dataProvider"),
        }
    
    
    def generate_film_doc(self) -> Document:
        flat_lines = flatten_for_rag({
            "blurb": self.metadata.get("blurb"),
        })
        page_content = "\n".join(flat_lines)
        return Document(
            page_content=page_content,
            metadata={
                "film_id": self.id,
                "title": self.metadata.get("title"),
                "year": self.metadata.get("year"),
                "country": self.metadata.get("country"),
                "production": self.metadata.get("production"),
                "language": self.metadata.get("language"),
                "dataProvider": self.metadata.get("dataProvider"),
                "video_file": str(self.video_file),
            }
        )
    

    def load_cuts(self):
        """Load CutSegments from cuts.json if available."""
        if self.cuts_json.exists():
            with open(self.cuts_json, "r") as f:
                cut_dicts = json.load(f)
            self.cut_segments = [CutSegment.from_dict(d) for d in cut_dicts]
            print(f"Loaded {len(self.cut_segments)} cuts from {self.cuts_json}")
            return True
        return False


    def detect_and_save_cuts(self, cut_detector):
        cuts = cut_detector.detect_scenes(str(self.video_file))
        segments = cut_detector.save_all_cuts(self.video_file, cuts, self.cuts_output_dir)
        with open(self.cuts_json, "w") as f:
            json.dump(segments, f, indent=2)
        for i, seg in enumerate(segments, start=1):
            cut_segment = CutSegment.from_dict(seg)
            self.cut_segments.append(cut_segment)


    def add_cut_segment(self, cut_segment: CutSegment):
        self.cut_segments.append(cut_segment)


    def load_cut_segments(self, segment_dicts: list[dict]):
        for i, seg in enumerate(segment_dicts, start=1):
            seg["cut_id"] = i
            cut_segment = CutSegment.from_dict(seg)
            self.add_cut_segment(cut_segment)


    def enrich_cut_keyframes(self, cut_id: int, keyframe_data: list[dict]):
        cut = next((c for c in self.cut_segments if c.cut_id == cut_id), None)
        if not cut:
            raise ValueError(f"Cut {cut_id} not found")
        for kf in keyframe_data:
            cut.add_keyframe(
                frame_id=kf["frame_id"],
                image_path=Path(kf["image_path"]),
                objects=kf.get("objects", []),
                shot_type=kf.get("shot_type")
            )

    def generate_transitions(self):
        self.transitions = []
        for i in range(len(self.cuts) - 1):
            self.transitions.append(Transition(self.cuts[i], self.cuts[i+1]))


class Transition:
    def __init__(self, from_cut: CutSegment, to_cut: CutSegment):
        self.film_id = from_cut.film_id
        self.from_cut = from_cut
        self.to_cut = to_cut
        self.id = str(uuid.uuid4())


    def generate_transition_doc(self) -> Document:
        prev_shot = self.from_cut.metadata.get("shot_type", "unknown shot")
        next_shot = self.to_cut.metadata.get("shot_type", "unknown shot")

        prev_objs_list = [obj.get("label") for obj in self.from_cut.metadata.get("objects", [])]
        next_objs_list = [obj.get("label") for obj in self.to_cut.metadata.get("objects", [])]

        prev_objs = ", ".join(prev_objs_list) or "something"
        next_objs = ", ".join(next_objs_list) or "something"

        text = f"A {prev_shot} of {prev_objs} cuts to a {next_shot} of {next_objs}."

        return Document(
            page_content=text,
            metadata={
                "transition_id": self.id,
                "film_id": self.film_id,
                "from_cut_id": self.from_cut.id,
                "to_cut_id": self.to_cut.id,
            }
        )