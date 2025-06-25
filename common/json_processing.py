from pathlib import Path
import json

def load_jsons(path):
    data = []
    for file in Path(path).rglob("*.json"):
        with open(file, "r") as f:
            film = json.load(f)
            data.append(film)
    return data