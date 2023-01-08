
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from models.lightning_modules import ImageClassifierLightningModule
from datasets.datasets import ImageDataset

def train_image_classifier(model: torch.nn.Module, train_dataset: ImageDataset, val_dataset: ImageDataset):
    assert len(train_dataset.labels) == len(val_dataset.labels)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    batches = int(np.ceil(len(train_dataset) / wandb.config.batch_size))

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    wandb_logger = wandb.WandbLogger()

    lightning_model = ImageClassifierLightningModule(model, len(train_dataset.labels))
    trainer = pl.Trainer(
        max_epochs=wandb.config.epochs,
        log_every_n_steps=5,
        logger=wandb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader, 
        val_dataloaders=val_data_loader,
    )

