#!/bin/bash

# matek
python -m tasks.vis.clusters_sweep \
    dataset=matek \
    training=matek_classifier_2023_04_11 \
    features=matek_classifier_2023_04_11 \
    clusters_highlight=15

# isic
python -m tasks.vis.clusters_sweep \
    dataset=isic \
    training=isic_classifier_2023_04_11 \
    features=isic_classifier_2023_04_11 \
    clusters_highlight=8

# retinopathy
python -m tasks.vis.clusters_sweep \
    dataset=retinopathy \
    training=retinopathy_classifier_2023_04_11 \
    features=retinopathy_classifier_2023_04_11 \
    clusters_highlight=5

# jurkat
python -m tasks.vis.clusters_sweep \
    dataset=jurkat \
    training=jurkat_classifier_2023_04_11 \
    features=jurkat_classifier_2023_04_11 \
    clusters_highlight=7

# cifar10
python -m tasks.vis.clusters_sweep \
    dataset=cifar10 \
    training=cifar10_classifier_2023_04_11 \
    features=cifar10_classifier_2023_04_11 \
    clusters_highlight=10
