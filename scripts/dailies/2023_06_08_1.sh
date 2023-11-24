#!/bin/bash

python -m tasks.vis.clusters_sweep \
    training.epochs=100 \
    training.weights.type=simclr \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=matek \
    clusters_highlight=15 \
    use_scan_weights=null

python -m tasks.vis.clusters_sweep \
    training.epochs=100 \
    training.weights.type=simclr \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=isic \
    clusters_highlight=8 \
    use_scan_weights=null

python -m tasks.vis.clusters_sweep \
    training.epochs=100 \
    training.weights.type=simclr \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=retinopathy \
    clusters_highlight=5 \
    use_scan_weights=null

python -m tasks.vis.clusters_sweep \
    training.epochs=100 \
    training.weights.type=simclr \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=jurkat \
    clusters_highlight=7 \
    use_scan_weights=null

python -m tasks.vis.clusters_sweep \
    training.epochs=100 \
    training.weights.type=simclr \
    training.weights.version=v1 \
    training.weights.freeze=true \
    dataset=cifar10 \
    clusters_highlight=10 \
    use_scan_weights=null
