#!/bin/bash

 python -m tasks.training.cold_paws \
        --multirun \
        training.weights.type=simclr \
        training.weights.version=v1 \
        training.weights.freeze=true \
        training.epochs=200 \
        use_scan_weights=false \
        dataset=cifar10,matek,isic,retinopathy,jurkat 
