from pathlib import Path

def get_film_title(path: str):
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

videos = find_film_files("datasets/")
for video in videos:
    print(video)
print(len(videos))

title = get_film_title(videos[786])
print(title)