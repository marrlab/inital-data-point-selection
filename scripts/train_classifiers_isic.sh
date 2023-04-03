#!/bin/bash

DATASET=isic
EPOCHS=50
LEARNING_RATE=0.001
WEIGHTS_PATH="outputs/2023-03-31/08-15-53/lightning_logs/z35miyut/checkpoints/epoch=995-step=157368-val_loss_ssl=3.99.ckpt"
FEATURES_PATH="outputs/2023-04-02/12-52-36/features.csv"
RUN_RANDOM_BASELINE=true
WEIGHTS_FREEZE_OPTIONS=(true false)
KMEANS_MODE_OPTIONS=("kmeans++" "kmeans")
KMEANS_CRITERIUM_OPTIONS=("closest" "furthest")

for i in $(seq 1 5); do
    for freeze_option in "${WEIGHTS_FREEZE_OPTIONS[@]}"; do
        # random baseline
        if [ $RUN_RANDOM_BASELINE ]; then
            python -m tasks.training.random_baseline dataset=$DATASET training.epochs=$EPOCHS training.learning_rate=$LEARNING_RATE \
                training.weights.freeze=$freeze_option training.weights.path=$WEIGHTS_PATH
        fi

        # badge sampling
        for kmeans_mode in "${KMEANS_MODE_OPTIONS[@]}"; do
            for kmeans_criterium in "${KMEANS_CRITERIUM_OPTIONS[@]}"; do
                python -m tasks.training.badge_sampling dataset=$DATASET training.epochs=$EPOCHS training.learning_rate=$LEARNING_RATE \
                    training.weights.freeze=$freeze_option training.weights.path=$WEIGHTS_PATH \
                    features.scaling=standard features.path=$FEATURES_PATH \
                    kmeans.mode=$kmeans_mode kmeans.criterium=$kmeans_criterium
            done
        done
    done
done
