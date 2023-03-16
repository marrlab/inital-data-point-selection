
import torch
import torchvision
import pytorch_lightning as pl
import numpy as np
from src.models.lightning_modules import ImageClassifierLightningModule
import wandb
from src.datasets.datasets import ImageDataset, get_dataset_class_by_name
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate

def train_image_classifier(model: torch.nn.Module, train_dataset: ImageDataset, val_dataset: ImageDataset):
    # TODO: revise
    assert len(train_dataset.labels) == len(val_dataset.labels)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    batches = int(np.ceil(len(train_dataset) / wandb.config.batch_size))

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=wandb.config.batch_size)
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
        max_epochs=wandb.config.epochs,
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


# TODO: try with other self-supervised models
def train_simclr():
    pl.seed_everything(wandb.config.seed)

    # creating datasets
    dataset_class = get_dataset_class_by_name(wandb.config.dataset)
    train_dataset = dataset_class('train')
    val_dataset = dataset_class('test')

    # we create a torchvision transformation for embedding the dataset after training
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (wandb.config.input_size, wandb.config.input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize['mean'],
            std=collate.imagenet_normalize['std'],
        )
    ])

    # transforming our datasets to lightly datasets
    train_dataset_lightly = LightlyDataset(
        input_dir=train_dataset.images_dir
    )

    val_dataset_lightly = LightlyDataset(
        input_dir=val_dataset.images_dir,
        transform=val_transforms
    )

    # augmentations for simclr
    collate_fn = SimCLRCollateFunction(
        input_size=wandb.config.input_size,
        vf_prob=0.5,
        rr_prob=0.5
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset_lightly,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=wandb.config.num_workers
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset_lightly,
        batch_size=wandb.config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=wandb.config.num_workers
    )

    # defining the model
    lightning_model = SimCLRModel(
        max_epochs=wandb.config.epochs,
        imagenet_weights=wandb.config.imagenet_weights,
    )

    # wandb connection (assumes wandb.init has been called before)
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.watch(lightning_model)

    trainer = pl.Trainer(
        max_epochs=wandb.config.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader,
        # TODO: figure this out
        # val_dataloaders=val_data_loader,
    )

    # saving the final model
    trainer.save_checkpoint(wandb.config.model_save_path, weights_only=True)
