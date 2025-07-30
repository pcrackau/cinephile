from pathlib import Path

def get_film_title(path: Path):
    file = Path(path)
    title = file.stem
    return title

def find_film_files(directory, extensions=None):
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    video_files = [
        file for file in Path(directory).rglob("*")
        if file.suffix.lower() in extensions
    ]
    return video_files

def collect_jpgs(directory, extensions=None):
    if extensions is None:
        extensions = [".jpg"]
    image_files = [
        file for file in Path(directory).rglob("*")
        if file.suffix.lower() in extensions
    ]
    return image_files

def find_film_metadata_pairs(directory, extensions=None, metadata_ext=".json"):
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    film_metadata_pairs = []

    for video_file in Path(directory).rglob("*"):
        if video_file.suffix.lower() not in extensions:
            continue

        candidate_json = video_file.with_suffix(metadata_ext)
        if candidate_json.exists():
            film_metadata_pairs.append((video_file, candidate_json))
        else:
            print(f"No metadata found for: {video_file.name}")

    return film_metadata_pairs