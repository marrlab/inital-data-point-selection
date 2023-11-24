#!/bin/bash

python -m tasks.vis.clusters_sweep \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=matek \
    use_scan_weights=null \
    clusters_highlight=15

python -m tasks.vis.clusters_sweep \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=isic \
    use_scan_weights=null \
    clusters_highlight=8

python -m tasks.vis.clusters_sweep \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=retinopathy \
    use_scan_weights=null \
    clusters_highlight=5

python -m tasks.vis.clusters_sweep \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=jurkat \
    use_scan_weights=null \
    clusters_highlight=7

python -m tasks.vis.clusters_sweep \
    training.epochs=75 \
    training.weights.type=swav \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=cifar10 \
    use_scan_weights=null \
    clusters_highlight=10
