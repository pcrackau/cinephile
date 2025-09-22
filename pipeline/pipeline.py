import os
import sys
sys.path.append(os.getcwd())
import argparse
import json

from ultralytics import YOLO
from KeyFrameDetector.key_frame_detector import keyframeDetection
from cutdetection import CutDetector
from pathlib import Path

from common.helper import find_films, pipeline_steps
from common.structs import CutSegment

from embedding import load_chroma
from identifyobjects import detect_objects
from shottypes import predict_shottype

from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import filter_complex_metadata


DATASETS = "datasets/"
#DATASETS = "datasets_2/"   # for testing purposes


def main(args : argparse.Namespace):
    steps = pipeline_steps(args.steps)
    
    films = find_films(DATASETS)
    for film in films:
        print(film.video_file)

    model_path = os.path.abspath("models/yolov8x.pt")
    model = YOLO(model_path)
    
    db = load_chroma()
    
    for film in films:
        print(f"Processing {film.video_file}")
        print(film.metadata)

        # mk directories so that step can be skipped
        keyframes_dir = Path("processing") / film.video_file.stem / "key_frames"
        cuts_output_dir = Path("processing") / film.video_file.stem / "cut_segments"
        cuts_output_dir.mkdir(parents=True, exist_ok=True)
        
        #if 1 in steps:
        ## Step 1: Cut detection, split films
        cut_detector = CutDetector(threshold=20.0)
        cut_dicts = cut_detector.save_all_cuts(film.video_file, film.cuts, cuts_output_dir)
        film.cuts = []
        for cut_info in cut_dicts:
            cut_segment = CutSegment(
                film_id=film.id,
                cut_file=cut_info["cut_file"],        # path to cut video
                start_time=cut_info["start_time"],
                end_time=cut_info["end_time"],
                duration=cut_info["duration"],
                start_frame=cut_info["start_frame"],
                end_frame=cut_info["end_frame"],
                fps=cut_info["fps"]
            )
            film.cuts.append(cut_segment)

        with open(f"processing/{film.video_file.stem}/cuts.json", "w") as f:
            json.dump(cut_dicts, f, indent=2)

        #if 2 in steps:
        ## Step 2: Key frame detection
        for cut in film.cuts:
            keyframes_dir = Path("processing") / film.video_file.stem / "key_frames" / cut.id
            keyframes_dir.mkdir(parents=True, exist_ok=True)
            keyframeDetection(
                source=str(cut.cut_file),
                dest=str(keyframes_dir),
                Thres=0.8,
                plotMetrics=False,
                verbose=False
            )

            keyframe = Path("processing") / film.video_file.stem / "key_frames" / cut.id / "keyFrames" / "keyframe1.jpg"
            objects, shot_type = None, None
            #if 3 in steps:
            ## Step 3: Object detection on keyframe
            objects = detect_objects(keyframe, model)

            #if 4 in steps:
            ## Step 4: Shot type detection on keyframe
            shot_type = predict_shottype(keyframe)

            cut.metadata["objects"] = objects
            cut.metadata["shot_type"] = shot_type
        film.generate_transitions()

    
    stored_jsons = Path("processing") / "docs"
    stored_jsons.mkdir(parents=True, exist_ok=True)

    docs = []

    # TODO refactor into docs.append(film.generate_film_doc()) etc once print not needed anymore
    for film in films:
        film_doc = film.generate_film_doc()
        print(film_doc)
        docs.append(film_doc)

        for cut in film.cuts:
            cut_doc = cut.generate_cut_doc()
            print(cut_doc)
            docs.append(cut_doc)

        for transition in film.transitions:
            trans_doc = transition.generate_transition_doc()
            print(trans_doc)
            docs.append(trans_doc)

    ## Embedding
    if docs:
        lc_docs = filter_complex_metadata(docs)
        db.add_documents(lc_docs)     


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set boolean flag to overwrite existing cut segments.')
    parser.add_argument('-s', '--steps',
                    required=False,
                    default="1,2",
                    help='Specify pipeline steps, either with e.g. 2-4 for steps 2,3,4 or separated by comma.'  )
    
    args : argparse.Namespace = parser.parse_args()
    main(args)