#!/bin/bash

for i in $(seq 1 5); do
    python -m tasks.training.badge_sampling \
        --multirun \
        training=classifier_base \
        training.weights.type=dino \
        training.weights.version=v2 \
        training.weights.freeze=true \
        training.oversample=true \
        use_scan_weights=false \
        TODO
        training.epochs= \
        kmeans.clusters=1,10,25,100 \
        kmeans.criterium=fps,closest,furthest,random,half_in_half \
        dataset=cifar10,matek,isic,retinopathy,jurkat
done
