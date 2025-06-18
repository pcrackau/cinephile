import os
import sys
sys.path.append(os.getcwd())
import argparse

from common.json_processing import load_jsons

PATH_FILMS = "../datasets"



def main(args : argparse.Namespace):

    film_jsons = load_jsons(PATH_FILMS)
    print(f"Loaded {len(film_jsons)} films.")
    print(film_jsons[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path to input-folder, if not hard-coded already.')
    parser.add_argument('-i', '--input_folder',
                    required=False,
                    help='Name of the dataset folder.')
    
    args : argparse.Namespace = parser.parse_args()
    main(args)