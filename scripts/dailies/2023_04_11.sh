#!/bin/bash

nohup bash -c ' \
./scripts/train_classifiers.sh \
    -d matek \
    -w "outputs/2023-04-06/12-19-01/lightning_logs/uq7vzdxh/checkpoints/epoch\=984-step\=27580-val_loss_ssl\=5.32.ckpt" \
    -f outputs/2023-04-09/12-50-29/features.csv \
&& \
./scripts/train_classifiers.sh \
    -d isic \
    -w "outputs/2023-04-05/23-04-29/lightning_logs/rbn9afqg/checkpoints/epoch\=999-step\=39000-val_loss_ssl\=5.33.ckpt" \
    -f outputs/2023-04-09/12-57-39/features.csv \
&& \
./scripts/train_classifiers.sh \
    -d retinopathy \
    -w "outputs/2023-04-05/18-49-26/lightning_logs/y52wl6b8/checkpoints/epoch\=974-step\=4875-val_loss_ssl\=5.04.ckpt" \
    -f outputs/2023-04-09/13-04-47/features.csv \
&& \
./scripts/train_classifiers.sh \
    -d jurkat \
    -w "outputs/2023-04-10/14-51-10/lightning_logs/vw8jsukl/checkpoints/epoch\=999-step\=12000-val_loss_ssl\=6.57.ckpt" \
    -f outputs/2023-04-10/23-01-42/features.csv \
&& \
./scripts/train_classifiers.sh \
    -d cifar10 \
    -w "outputs/2023-04-09/18-03-54/lightning_logs/1lcvwh2w/checkpoints/epoch\=989-step\=11880-val_loss_ssl\=7.58.ckpt" \
    -f outputs/2023-04-11/10-50-20/features.csv \
' &>> train_classifiers.log &
