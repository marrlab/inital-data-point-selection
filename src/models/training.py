
import os
import torch
import torchvision
import lightning.pytorch as pl
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models.lightning_modules import ImageClassifierLightningModule, SimCLRModel
from src.utils.utils import get_the_best_accelerator
import wandb
from src.datasets.datasets import ImageDataset, get_dataset_class_by_name
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig
from hydra.utils import get_original_cwd


def train_image_classifier(model: torch.nn.Module, train_dataset: ImageDataset, val_dataset: ImageDataset, cfg: DictConfig):
    assert len(train_dataset.labels) == len(val_dataset.labels)
    assert len(train_dataset.classes) == len(val_dataset.classes)

    # setting up oversampling if chosen
    train_data_loader = None
    if cfg.training.oversample:
        class_counts = defaultdict(int)
        for i in range(len(train_dataset)):
            class_counts[train_dataset[i]['class']] += 1

        weights = [1 / class_counts[train_dataset[i]['class']]
                   for i in range(len(train_dataset))]

        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_dataset), replacement=True)
        train_data_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, sampler=sampler)
    else:
        train_data_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    val_data_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    wandb_logger = pl.loggers.WandbLogger()

    class LogPredictionSamplesCallback(pl.Callback):
        def on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            """Called when the validation batch ends."""

            # `outputs` comes from `LightningModule.validation_step`
            # which corresponds to our model predictions in this case

            # Let's log n sample image predictions from the first batch
            if batch_idx == 0:
                n = 16
                images = list(batch['image'][:n])
                captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}'
                            for y_i, y_pred in zip(batch['label'][:n], outputs['pred_labels'][:n])]

                # Option 1: log images with `WandbLogger.log_image`
                wandb_logger.log_image(
                    key='validation_sample_images',
                    images=images,
                    caption=captions)

    lightning_model = ImageClassifierLightningModule(
        model,
        train_dataset.get_number_of_labels(),
        train_dataset.get_number_of_classes(),
        train_dataset.label_to_class_mapping.copy(),
        train_dataset.class_to_label_mapping.copy(),
        cfg
    )
    wandb_logger.watch(lightning_model)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=20,
        logger=wandb_logger,
        log_every_n_steps=5,
        accelerator=get_the_best_accelerator(),
        devices=1,
        callbacks=[
            # TODO: comment out for debugging
            LogPredictionSamplesCallback(),
            # ModelCheckpoint(mode='max', monitor='val_f1_macro_epoch_end',
            #                 save_top_k=1, filename='{epoch}-{step}-{val_f1_macro_epoch_end:.2f}'),
            # ModelCheckpoint(every_n_epochs=10),
            LearningRateMonitor('epoch'),
        ]
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )

    # trainer.save_checkpoint(cfg.training.model_save_path)
