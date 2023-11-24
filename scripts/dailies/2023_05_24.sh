#!/bin/bash

python -m tasks.training.train_swav dataset=cifar10

python -m tasks.training.train_swav dataset=matek

python -m tasks.training.train_swav dataset=isic

python -m tasks.training.train_swav dataset=retinopathy

python -m tasks.training.train_swav dataset=jurkat
