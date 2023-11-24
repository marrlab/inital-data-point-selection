#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [--push | --pull]"
    exit 1
fi

operation=$1

if [[ "$operation" == "--push" ]]; then
    rclone copy --progress src/models/data/ google-drive:master\ thesis/models/data --exclude=precomputed_features/**
elif [[ "$operation" == "--pull" ]]; then
    rclone copy --progress google-drive:master\ thesis/models/data src/models/data
else
    echo "Invalid operation: $operation"
    exit 1
fi
