
#!/bin/bash

./scripts/train_badge_sampling.sh \
    -d cifar10 \
    -t cifar10_classifier_2023_04_11 \
    -f cifar10_classifier_2023_04_11 \
    -c 1 \
    -a random,closest,furthest,half_in_half

./scripts/train_badge_sampling.sh \
    -d matek \
    -t matek_classifier_2023_04_11 \
    -f matek_classifier_2023_04_11 \
    -c 1 \
    -a random,closest,furthest,half_in_half

./scripts/train_badge_sampling.sh \
    -d isic \
    -t isic_classifier_2023_04_11 \
    -f isic_classifier_2023_04_11 \
    -c 1 \
    -a random,closest,furthest,half_in_half

./scripts/train_badge_sampling.sh \
    -d jurkat \
    -t jurkat_classifier_2023_04_11 \
    -f jurkat_classifier_2023_04_11 \
    -c 1 \
    -a random,closest,furthest,half_in_half

./scripts/train_badge_sampling.sh \
    -d retinopathy \
    -t retinopathy_classifier_2023_04_11 \
    -f retinopathy_classifier_2023_04_11 \
    -c 1 \
    -a random,closest,furthest,half_in_half
