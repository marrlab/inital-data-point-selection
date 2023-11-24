#!/bin/bash

# matek
python -m tasks.training.train_dino \
    dataset=matek \
    training.batch_size=1024 \
    training.learning_rate=1e-1 \
    training.epochs=400

# TODO
# isic
python -m tasks.training.train_dino \
    dataset=isic \
    training.batch_size=1024 \
    training.learning_rate=1e-2 \
    training.epochs=400

# TODO
# retinopathy
python -m tasks.training.train_dino \
    dataset=retinopathy \
    training.batch_size=1024 \
    training.learning_rate=1e-1 \
    training.epochs=1000

# jurkat
python -m tasks.training.train_dino \
    dataset=jurkat \
    training.batch_size=2048 \
    training.learning_rate=1e-1 \
    training.epochs=400

# cifar10
python -m tasks.training.train_dino \
    dataset=cifar10 \
    training.batch_size=4096 \
    training.learning_rate=1e-1 \
    training.epochs=400


