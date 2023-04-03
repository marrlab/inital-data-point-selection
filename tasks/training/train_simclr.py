
import hydra
import lightning.pytorch as pl
import torch
from lightly.data import LightlyDataset, SimCLRCollateFunction
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from src.datasets.datasets import get_dataset_class_by_name
from src.models.lightning_modules import SimCLRModel
from src.utils.wandb import init_run


# TODO: try with other self-supervised models
@hydra.main(version_base=None, config_path='../../conf', config_name='train_simclr')
def main(cfg: DictConfig):
    init_run(cfg)

    pl.seed_everything(cfg.training.seed)

    # creating datasets
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class('train')
    val_dataset = dataset_class('test')

    train_dataset_lightly = LightlyDataset(
        input_dir=train_dataset.images_dir
    )

    val_dataset_lightly = LightlyDataset(
        input_dir=val_dataset.images_dir,
        # transform=val_transforms
    )

    # augmentations for simclr
    collate_fn = SimCLRCollateFunction(
        input_size=cfg.training.input_size,
        vf_prob=0.5,
        rr_prob=0.5
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset_lightly,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=cfg.training.num_workers
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset_lightly,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=cfg.training.num_workers
    )

    # defining the model
    lightning_model = SimCLRModel(
        max_epochs=cfg.training.epochs,
        imagenet_weights=cfg.training.weights.type == 'imagenet',
    )

    # wandb connection (assumes wandb.init has been called before)
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.watch(lightning_model)

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[
            ModelCheckpoint(mode='min', monitor='val_loss_ssl',
                            save_top_k=3, filename='{epoch}-{step}-{val_loss_ssl:.2f}'),
            ModelCheckpoint(every_n_epochs=50),
            LearningRateMonitor('epoch'),
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader
    )

    # saving the final model
    trainer.save_checkpoint(cfg.training.model_save_path)


if __name__ == '__main__':
    main()
