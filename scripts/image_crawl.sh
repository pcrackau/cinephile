#!/bin/bash

BASE_URL="https://shots.filmschoolrejects.com/shot-types"
SHOT_TYPES=("long-shot" "medium-shot" "medium-long-shot" "close-up" "extreme-close-up")
MAX_PAGES=20

mkdir -p ../images

for shot in "${SHOT_TYPES[@]}"; do
    echo "Processing shot type: $shot"
    mkdir -p "../images/$shot"

    for ((page=1; page<=MAX_PAGES; page++)); do
        if [ "$page" -eq 1 ]; then
            URL="$BASE_URL/$shot/"
        else
            URL="$BASE_URL/$shot/page/$page/"
        fi

        echo "Fetching $URL"
        HTML=$(curl -s "$URL")

        # Extract .jpg image URLs
        IMG_URLS=$(echo "$HTML" | grep -oE 'https://[^"]+\.jpg')

        if [ -z "$IMG_URLS" ]; then
            echo "No images found. Stopping at page $page."
            break
        fi

        while IFS= read -r img_url; do
            filename=$(basename "$img_url")
            if [ ! -f "images/$shot/$filename" ]; then
                echo "Downloading $filename"
                curl -s "$img_url" -o "images/$shot/$filename"
            else
                echo "Already downloaded: $filename"
            fi
        done <<< "$IMG_URLS"
    done
done

# Remove logo image files
find finetune_data/ -type f -name '*cropped-ops-new-sq*' -exec rm {} \;

echo "Done. Images stored in ./images/<shot-type>/"
