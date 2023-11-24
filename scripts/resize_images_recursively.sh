#!/bin/bash

# Check if the folder path and target size are provided as command line arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <folder_path> <target_size>"
    exit 1
fi

folder_path="$1"
target_size="$2"

# Find all image files recursively in the specified folder
find "$folder_path" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tiff" \) -print0 |
while IFS= read -r -d '' image; do
    echo "Resizing image: $image"
    # Resize the image using ImageMagick's convert command
    convert "$image" -resize "$target_size" "$image"
done
