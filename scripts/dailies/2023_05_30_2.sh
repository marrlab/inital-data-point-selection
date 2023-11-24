#!/bin/bash

python -m tasks.vis.umap_features \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=matek
    

python -m tasks.vis.umap_features \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=isic


python -m tasks.vis.umap_features \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=retinopathy
    

python -m tasks.vis.umap_features \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=jurkat 
    

python -m tasks.vis.umap_features \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=cifar10
    
