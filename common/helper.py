import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from common.structs import Film


def find_film_files(directory, extensions=None):
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    video_files = [
        file for file in Path(directory).rglob("*")
        if file.suffix.lower() in extensions
    ]
    return video_files


def find_films(directory, extensions=None, metadata_ext=".json") -> list[Film]:
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

    films = []
    for video_file in Path(directory).rglob("*"):
        if video_file.suffix.lower() not in extensions:
            continue

        candidate_json = video_file.with_suffix(metadata_ext)
        if candidate_json.exists():
            film = Film(video_file, candidate_json)
            films.append(film)
        else:
            print(f"No metadata found for: {video_file.name}")

    return films


def pipeline_steps(steps: str):
    step_lst = []
    step_chars = list(steps.replace(',',''))
    if step_chars.__contains__('-'):
        step_lst = list(range(int(step_chars[0]),int(step_chars[-1])+1))
    else:
        step_lst = list(map(int,step_chars))
    return step_lst


