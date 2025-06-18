import os
import sys
sys.path.append(os.getcwd())
import argparse

from common.json_processing import load_jsons

import time
import requests

PATH_FILMS = "../datasets"      # relative path
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def query_ollama(prompt, model=MODEL_NAME):     # check which models available in command line: ollama list
    start_time = time.time()
    url = OLLAMA_URL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        #"options": {"num_predict": 128}
    }

    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    result = response.json()
    if "response" not in result:
        raise KeyError(f"Missing 'response' in: {result}")

    duration = time.time() - start_time
    print(f"Query time {duration:.5f}s")

    return result["response"]


# TODO move ollama call to own file
# TODO integrate json into responses
def main(args : argparse.Namespace):

    film_jsons = load_jsons(PATH_FILMS)
    print(f"Loaded {len(film_jsons)} films.")
    print(film_jsons[0])

    # Example usage
    try:
        reply = query_ollama("What is the plot of Western Front?")
        print(reply)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path to input-folder, if not hard-coded already.')
    parser.add_argument('-i', '--input_folder',
                    required=False,
                    help='Name of the dataset folder.')
    
    args : argparse.Namespace = parser.parse_args()
    main(args)