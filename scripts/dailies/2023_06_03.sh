#!/bin/bash

for i in $(seq 1 5); do
    python -m tasks.training.badge_sampling --multirun \
        training.epochs=100 \
        training.weights.type=simclr \
        training.weights.version=v1 \
        training.weights.freeze=true \
        kmeans.mode=kmeans \
        kmeans.criterium=closest,furthest,random,half_in_half,fps \
        kmeans.clusters=1,2,5,7,8,10,15,20,25,50,100 \
        dataset=matek,isic,retinopathy,jurkat,cifar10
done
