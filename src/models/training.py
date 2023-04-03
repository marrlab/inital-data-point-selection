
import os
import torch
import torchvision
import lightning.pytorch as pl
import numpy as np
from src.models.lightning_modules import ImageClassifierLightningModule, SimCLRModel
from src.utils.utils import have_models_same_weights
import wandb
from src.datasets.datasets import ImageDataset, get_dataset_class_by_name
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

def train_image_classifier(model: torch.nn.Module, train_dataset: ImageDataset, val_dataset: ImageDataset, cfg: DictConfig):
    # TODO: revise
    assert len(train_dataset.labels) == len(val_dataset.labels)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    batches = int(np.ceil(len(train_dataset) / cfg.training.batch_size))

    val_data_loader = torch.utils.data.DataLoader(
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
                            for y_i, y_pred in zip(batch['label'][:n], outputs['preds'][:n])]

                # Option 1: log images with `WandbLogger.log_image`
                wandb_logger.log_image(
                    key='validation_sample_images',
                    images=images,
                    caption=captions)

    lightning_model = ImageClassifierLightningModule(
        model, len(train_dataset.labels), cfg, labels_text=train_dataset.labels_text)
    wandb_logger.watch(lightning_model)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        log_every_n_steps=5,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[
            # TODO: commented for debugging
            # LogPredictionSamplesCallback(),
            ModelCheckpoint(mode='min', monitor='val_loss_epoch_end',
                            save_top_k=3, filename='{epoch}-{step}-{val_loss_ssl:.2f}'),
            ModelCheckpoint(every_n_epochs=10),
            LearningRateMonitor('epoch'),
        ]
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )

    trainer.save_checkpoint(cfg.training.model_save_path)
