#!/bin/bash

python -m tasks.inference.extract_features \
    --multirun \
    training.weights.type=dino \
    training.weights.version=v2 \
    use_scan_weights=false \
    dataset=matek,isic,retinopathy,jurkat,cifar10 
