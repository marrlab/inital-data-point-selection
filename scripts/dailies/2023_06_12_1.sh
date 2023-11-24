#!/bin/bash

python -m tasks.inference.extract_features \
    --multirun \
    training.weights.type=simclr \
    training.weights.version=v1 \
    use_scan_weights=true \
    dataset=matek,isic,retinopathy,jurkat,cifar10 
