#!/bin/bash

python -m tasks.training.train_scan \
    --multirun \
    training.weights.type=simclr \
    training.weights.version=v1 \
    training.weights.freeze=false \
    training.epochs=400 \
    training.learning_rate=1e-4 \
    scan.dropout=0.5 \
    dataset=matek,isic,retinopathy,jurkat,cifar10 
