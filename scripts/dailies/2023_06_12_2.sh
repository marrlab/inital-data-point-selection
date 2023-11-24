#!/bin/bash

for i in $(seq 1 5); do
    python -m tasks.training.badge_sampling \
        --multirun \
        training=classifier_all_samples \
        training.weights.type=simclr \
        training.weights.version=v1 \
        training.weights.freeze=true \
        training.oversample=true \
        training.epochs=200 \
        kmeans=no_clustering \
        use_scan_weights=false,true \
        dataset=matek,isic,retinopathy,jurkat,cifar10
done
