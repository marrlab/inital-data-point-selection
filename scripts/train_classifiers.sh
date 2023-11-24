#!/bin/bash

# Define default values
weights_path=""
features_path=""
dataset=""
runs=5

# Print help message
function print_help {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "    -w|--weights-path   Path to weights file (mandatory)"
    echo "    -f|--features-path  Path to features file (mandatory)"
    echo "    -d|--dataset        Dataset name (mandatory)"
    echo "    -r|--runs           Number of runs (default: 5)"
    echo "    -h|--help           Print this help message"
    exit 0
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--weights-path)
            weights_path="$2"
            shift 2
            ;;
        -f|--features-path)
            features_path="$2"
            shift 2
            ;;
        -d|--dataset)
            dataset="$2"
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
if [ -z "$weights_path" ] || [ -z "$features_path" ] || [ -z "$dataset" ]; then
    echo "Error: Mandatory arguments not provided"
    print_help
fi

# Do something with the arguments
echo "Weights path: $weights_path"
echo "Features path: $features_path"
echo "Dataset: $dataset"
echo "Number of runs: $runs"

# The actual script
RUN_RANDOM_BASELINE=true
WEIGHTS_FREEZE_OPTIONS=(true)
KMEANS_MODE_OPTIONS=("kmeans++" "kmeans")
KMEANS_CRITERIUM_OPTIONS=("closest" "furthest")

for i in $(seq 1 $runs); do
    for freeze_option in "${WEIGHTS_FREEZE_OPTIONS[@]}"; do
        # random baseline
        if [ $RUN_RANDOM_BASELINE ]; then
            python -m tasks.training.random_baseline dataset=$dataset \
                training.weights.freeze=$freeze_option training.weights.path=$weights_path
        fi

        # badge sampling
        for kmeans_mode in "${KMEANS_MODE_OPTIONS[@]}"; do
            for kmeans_criterium in "${KMEANS_CRITERIUM_OPTIONS[@]}"; do
                python -m tasks.training.badge_sampling dataset=$dataset \
                    training.weights.freeze=$freeze_option training.weights.path=$weights_path \
                    features.scaling=standard features.path=$features_path \
                    kmeans.mode=$kmeans_mode kmeans.criterium=$kmeans_criterium
            done
        done
    done
done
