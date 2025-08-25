import os
import sys
sys.path.append(os.getcwd())
import argparse
import json

from ultralytics import YOLO
from KeyFrameDetector.key_frame_detector import keyframeDetection
from cutdetection import CutDetector
from pathlib import Path

from common.helper import find_films, collect_jpgs, find_film_files

from embedding import embed_and_store, load_or_create_chroma, generate_docs, load_chroma
from identifyobjects import detect_objects
from shottypes import predict_shottype

#DATASETS = "datasets/"
DATASETS = "datasets_2/"
LANGUAGE = "en"


def main(args : argparse.Namespace):
    overwrite_flag = args.overwrite
    
    films = find_films(DATASETS)
    for film in films:

        print(film.video_path)


    model_path = os.path.abspath("models/yolov8x.pt")
    model = YOLO(model_path)
    db = load_chroma()
    docs = []

    
    for film in films:
    #for film_path in film_path_lst:
        print(f"Processing {film.video_path}")
        print(film.metadata_path)
"""        
        with open(metadata_path, "r") as f:
            film_metadata = json.load(f)

        ## Cut detection, split films
        cut_detector = CutDetector(threshold=20.0)
        cuts = cut_detector.detect_scenes(str(film_path))
        cuts_output_dir = Path("processing") / film_path.stem / "cut_segments"
        cuts_output_dir.mkdir(parents=True, exist_ok=True)
        cut_detector.save_all_cuts(film_path, cuts, cuts_output_dir)

        # TODO add timestamps of cut segments to metadata

        ## Key frame detection
        keyframes_dir = Path("processing") / film_path.stem / "key_frames"

        keyframeDetection(
            source=film_path,
            dest=str(keyframes_dir),
            Thres=0.8,   
            plotMetrics=False,     
            verbose=False      
        )

        keyframe_lst = collect_jpgs(keyframes_dir)

        for keyframe in keyframe_lst:

            ## Object detection on keyframe
            objects = detect_objects(keyframe, model)

            ## Shot type detection on keyframe
            shot_type = predict_shottype(keyframe)
            frame_info = {
                "filename": keyframe.name,
                "shot_type": shot_type,
                "objects": objects
            }

            ## json for each keyframe
            output_path = keyframe.with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(frame_info, f, indent=2)

            new_doc = generate_docs(metadata_path, output_path)
            docs.extend(new_doc)

    ## Embedding
    if docs:
        db.add_documents(docs)     
    db.persist()
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set boolean flag to overwrite existing cut segments.')
    parser.add_argument('-o', '--overwrite',
                    required=False,
                    default=True,
                    help='Name of the dataset folder.')
    
    args : argparse.Namespace = parser.parse_args()
    main(args)