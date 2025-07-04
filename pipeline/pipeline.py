import os
import sys
sys.path.append(os.getcwd())

from cutdetection import CutDetector
from pathlib import Path

from common.embedding import embed_and_store
from common.helper import get_film_title, find_film_files

DATASETS = "datasets/"
# TODO: 
# 0. Preprocessing
# 1. Cut detection -> separate scenes
# 2. Object detection -> from key frame of scene
# 3. Shot type detection -> from key frame
# 4. Embed all info in Chroma-DB (Persisting)


film_path_lst = find_film_files(DATASETS)   # has videoformat suffix

for film_path in film_path_lst:
    pass
    # Cut detection TODO: create folder separate from datasets with film-title

film_path = film_path_lst[786]    
print(film_path)
cut_detector = CutDetector(threshold=20.0)
cuts = cut_detector.detect_scenes(str(film_path))
print("Cuts:", cuts)

film_title = get_film_title(film_path)

output_dir = Path("processing") / film_path.stem / "cut_segments"
output_dir.mkdir(parents=True, exist_ok=True)
cut_detector.save_all_cuts(film_path, cuts, output_dir)