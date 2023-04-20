#!/bin/bash

# Define default values
training=""
features=""
dataset=""
clusters=""
runs=5

# Print help message
function print_help {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "    -t|--training       Training config (mandatory)"
    echo "    -f|--features       Features config (mandatory)"
    echo "    -d|--dataset        Dataset name (mandatory)"
    echo "    -c|--clusters       Cluster options (mandatory)"
    echo "    -r|--runs           Number of runs (default: 5)"
    echo "    -h|--help           Print this help message"
    exit 0
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--training)
            training="$2"
            shift 2
            ;;
        -f|--features)
            features="$2"
            shift 2
            ;;
        -d|--dataset)
            dataset="$2"
            shift 2
            ;;
        -c|--clusters)
            clusters="$2"
            shift 2
            ;;
        -r|--runs)
            runs="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Invalid option: $1"
            print_help
            ;;
    esac
done

# Check mandatory arguments
if [ -z "$training" ] || [ -z "$features" ] || [ -z "$dataset" ] || [ -z "$clusters" ]; then
    echo "Error: Mandatory arguments not provided"
    print_help
fi

# Do something with the arguments
echo "Training config: $training"
echo "Features config: $features"
echo "Dataset: $dataset"
echo "Cluster options: $clusters"
echo "Number of runs: $runs"

# The actual script
for i in $(seq 1 $runs); do
    # random baseline
    python -m tasks.training.badge_sampling \
        dataset=$dataset \
        training=$training \
        features=$features \
        kmeans.clusters=1 \
        kmeans.mode=kmeans++ \
        kmeans.criterium=random

    # badge sampling
    python -m tasks.training.badge_sampling \
        --multirun \
        dataset=$dataset \
        training=$training \
        features=$features \
        kmeans.clusters=$clusters \
        kmeans.mode=kmeans \
        kmeans.criterium=random,closest,furthest,half_in_half
done