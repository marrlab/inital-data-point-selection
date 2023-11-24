#!/bin/bash

for i in $(seq 1 5); do
    python -m tasks.training.badge_sampling \
        --multirun \
        training=classifier_base \
        training.weights.type=simclr \
        training.weights.version=v1 \
        training.weights.freeze=true \
        training.oversample=true \
        training.epochs=200 \
        kmeans.clusters=1 \
        kmeans.criterium=random \
        use_scan_weights=true \
        dataset=matek,isic,retinopathy,jurkat,cifar10
done
