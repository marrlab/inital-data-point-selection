#!/bin/bash

for i in $(seq 1 5); do
    python -m tasks.training.simclr_fps_cosine \
        --multirun \
        training.weights.type=simclr \
        training.weights.version=v1 \
        training.weights.freeze=true \
        training.epochs=200 \
        use_scan_weights=false \
        dataset=cifar10,matek,isic,retinopathy,jurkat 
done
