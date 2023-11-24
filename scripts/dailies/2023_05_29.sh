#!/bin/bash

for i in $(seq 1 4); do
    python -m tasks.training.badge_sampling_matek --multirun \
        training.epochs=75 \
        training.weights.type=swav \
        training.weights.version=v1 \
        training.weights.freeze=true \
        kmeans.mode=kmeans \
        kmeans.criterium=closest,furthest,random,half_in_half,fps \
        kmeans.clusters=1,2,5,7,8,10,15,20,25,50,100 \
        dataset=matek,retinopathy,jurkat

    python -m tasks.training.badge_sampling --multirun \
        training.epochs=75 \
        training.weights.type=swav \
        training.weights.version=v1 \
        training.weights.freeze=true \
        kmeans.mode=kmeans \
        kmeans.criterium=closest,furthest,random,half_in_half,fps \
        kmeans.clusters=1,2,5,7,8,10,15,20,25,50,100 \
        dataset=cifar10,isic
done

python -m tasks.training.badge_sampling_matek --multirun \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    kmeans.mode=kmeans \
    kmeans.criterium=closest,furthest,random,half_in_half,fps \
    kmeans.clusters=1,2,5,7,8,10,15,20,25,50,100 \
    dataset=matek,retinopathy,jurkat
