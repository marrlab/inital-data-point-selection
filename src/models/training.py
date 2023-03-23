
import torch
import torchvision
import lightning.pytorch as pl
import numpy as np
from src.models.lightning_modules import ImageClassifierLightningModule
import wandb
from src.datasets.datasets import ImageDataset, get_dataset_class_by_name
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from omegaconf import DictConfig

def train_image_classifier(model: torch.nn.Module, train_dataset: ImageDataset, val_dataset: ImageDataset, cfg: DictConfig):
    # TODO: revise
    assert len(train_dataset.labels) == len(val_dataset.labels)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    batches = int(np.ceil(len(train_dataset) / cfg.training.batch_size))

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.training.batch_size)
    wandb_logger = pl.loggers.WandbLogger()

    class LogPredictionSamplesCallback(pl.Callback):
        def on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            """Called when the validation batch ends."""

            # `outputs` comes from `LightningModule.validation_step`
            # which corresponds to our model predictions in this case

            # Let's log 20 sample image predictions from the first batch
            if batch_idx == 0:
                n = 8
                images = [img for img in batch['image'][:n]]
                captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}'
                            for y_i, y_pred in zip(batch['label'][:n], outputs['labels'][:n])]

                # Option 1: log images with `WandbLogger.log_image`
                wandb_logger.log_image(
                    key='validation_sample_images',
                    images=images,
                    caption=captions)

    lightning_model = ImageClassifierLightningModule(
        model, len(train_dataset.labels), labels_text=train_dataset.labels_text)
    wandb_logger.watch(lightning_model)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        log_every_n_steps=5,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[LogPredictionSamplesCallback()]
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )
