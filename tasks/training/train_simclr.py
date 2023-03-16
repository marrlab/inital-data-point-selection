
import sys
import torch
import wandb
import torchvision
import numpy as np
import pytorch_lightning as pl
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from src.models.lightning_modules import ImageClassifierLightningModule, SimCLRModel
from src.datasets.datasets import ImageDataset
from src.datasets.datasets import get_dataset_class_by_name
from src.utils.utils import load_yaml_as_dict, load_yaml_as_obj


# TODO: try with other self-supervised models
def main():
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


if __name__ == '__main__':
    task_name = sys.argv[1]
    config_path = sys.argv[2] 

    config = load_yaml_as_dict(config_path)

    if task_name == 'train_simclr':
        wandb.init(project='train-simclr', config=config)
        main()
    else:
        raise ValueError(f'unknown task name: {task_name}')

    # wandb wrapping-up
    wandb.finish()
