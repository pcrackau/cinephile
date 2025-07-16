import os
import sys
sys.path.append(os.getcwd())

from ultralytics import YOLO

from pathlib import Path
from common.helper import get_film_title, find_film_files

DATASETS = "datasets/"

film_path_lst = find_film_files(DATASETS)
film_path = film_path_lst[786]   
frames_output_dir = Path("processing") / film_path.stem / "key_frames"

model = YOLO("yolov8x.pt")
results = model(frames_output_dir / "keyFrames" / "keyframe8.jpg")
for result in results:
    #print(result.boxes)
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = results[0].names[class_id]
        confidence = float(box.conf[0])
        print(f"Detected: {label} ({confidence:.2%})")

def detect_objects(image):
    results = model(image)
    for result in results:
        print(result.boxes)


# finetune