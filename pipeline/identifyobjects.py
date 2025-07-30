import os
import sys
sys.path.append(os.getcwd())

from ultralytics import YOLO

from pathlib import Path
from common.helper import get_film_title, find_film_files

# TODO yolo only detects "person", expand with fairface or deepface
def detect_objects(image, model):
    results = model(image)
    objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]
            confidence = float(box.conf[0])
            print(f"Detected: {label} ({confidence:.2%})")
            print(result.boxes)
            bbox = [float(x) for x in box.xyxy[0].tolist()]
            objects.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox
            })
    
    return objects


# finetune