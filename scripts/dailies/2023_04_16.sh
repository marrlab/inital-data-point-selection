#!/bin/bash

nohup bash -c ' \
for i in $(seq 1 5); do \
    python -m tasks.training.badge_sampling \
              --multirun \
              dataset=cifar10 \
              features=cifar10_classifier_2023_04_11 \
              training=cifar10_classifier_2023_04_11 \
              kmeans.clusters=2,5,10,15,20,25,50,100 \
              kmeans.mode=kmeans \
              kmeans.criterium=random,closest,furthest \
    && \
    python -m tasks.training.badge_sampling \
              dataset=cifar10 \
              features=cifar10_classifier_2023_04_11 \
              training=cifar10_classifier_2023_04_11 \
              kmeans.clusters=1 \
              kmeans.mode=kmeans++ \
              kmeans.criterium=random \
done \
&& \
for i in $(seq 1 5); do \
    python -m tasks.training.badge_sampling \
              --multirun \
              dataset=matek \
              features=matek_classifier_2023_04_11 \
              training=matek_classifier_2023_04_11 \
              kmeans.clusters=2,5,10,15,20,25,50,100 \
              kmeans.mode=kmeans \
              kmeans.criterium=random,closest,furthest \
    && \
    python -m tasks.training.badge_sampling \
              dataset=matek \
              features=matek_classifier_2023_04_11 \
              training=matek_classifier_2023_04_11 \
              kmeans.clusters=1 \
              kmeans.mode=kmeans++ \
              kmeans.criterium=random \
done \
&& \
for i in $(seq 1 5); do \
    python -m tasks.training.badge_sampling \
              --multirun \
              dataset=isic \
              features=isic_classifier_2023_04_11 \
              training=isic_classifier_2023_04_11 \
              kmeans.clusters=2,5,8,10,15,20,25,50,100 \
              kmeans.mode=kmeans \
              kmeans.criterium=random,closest,furthest \
    && \
    python -m tasks.training.badge_sampling \
              dataset=isic \
              features=isic_classifier_2023_04_11 \
              training=isic_classifier_2023_04_11 \
              kmeans.clusters=1 \
              kmeans.mode=kmeans++ \
              kmeans.criterium=random \
done \
&& \
for i in $(seq 1 5); do \
    python -m tasks.training.badge_sampling \
              --multirun \
              dataset=retinopathy \
              features=retinopathy_classifier_2023_04_11 \
              training=retinopathy_classifier_2023_04_11 \
              kmeans.clusters=2,5,10,15,20,25,50,100 \
              kmeans.mode=kmeans \
              kmeans.criterium=random,closest,furthest \
    && \
    python -m tasks.training.badge_sampling \
              dataset=retinopathy \
              features=retinopathy_classifier_2023_04_11 \
              training=retinopathy_classifier_2023_04_11 \
              kmeans.clusters=1 \
              kmeans.mode=kmeans++ \
              kmeans.criterium=random \
done \
&& \
for i in $(seq 1 5); do \
    python -m tasks.training.badge_sampling \
              --multirun \
              dataset=jurkat \
              features=jurkat_classifier_2023_04_11 \
              training=jurkat_classifier_2023_04_11 \
              kmeans.clusters=2,5,7,10,15,20,25,50,100 \
              kmeans.mode=kmeans \
              kmeans.criterium=random,closest,furthest \
    && \
    python -m tasks.training.badge_sampling \
              dataset=jurkat \
              features=jurkat_classifier_2023_04_11 \
              training=jurkat_classifier_2023_04_11 \
              kmeans.clusters=1 \
              kmeans.mode=kmeans++ \
              kmeans.criterium=random \
done \
' &>> train_classifiers.log &
