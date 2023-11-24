
import hydra
import lightning.pytorch as pl
import torch
import wandb
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from src.datasets.datasets import get_dataset_class_by_name, NeighborsDataset
from src.models.classifiers import get_ssl_preprocess, get_ssl_transform
from src.utils.wandb import init_run
from src.utils.utils import get_the_best_accelerator
from src.utils.utils import get_cpu_count
from src.models.build_data_path import get_neighbors_path
from src.models.scan import SCAN


@hydra.main(version_base=None, config_path='../../conf', config_name='train_scan')
def main(cfg: DictConfig):
    assert cfg.training.weights.type == 'simclr', \
        'other ssl methods not yet supported'

    init_run(cfg)
    pl.seed_everything(cfg.training.seed)

    # fetching the augmentations
    transform = get_ssl_transform(cfg)    

    # creating datasets
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)

    train_dataset = dataset_class(
        split='train', 
        preprocess=torchvision.transforms.Resize(cfg.dataset.input_size, antialias=True),
        transform=transform, 
    )
    train_dataset = NeighborsDataset(train_dataset, get_neighbors_path(cfg))
    # val_dataset = dataset_class('val')

    # setting up the dataloader(s)
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=get_cpu_count()
    )

    # defining the model
    lightning_model = SCAN(cfg)
    lightning_model.backbone_from_ssl()

    if cfg.training.weights.freeze:
        lightning_model.freeze_backbone()

    # wandb connection (assumes wandb.init has been called before)
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.watch(lightning_model)

    # visualizations
    class LogImageSampleCallback(pl.Callback):
        def on_train_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if batch_idx == 0:
                self._common_batch_end('train', outputs, batch)

        def _common_batch_end(
                self, step, outputs, batch):
            max_n = 8
            n = min(len(batch['anchor_image']), max_n)

            images = []
            captions = []
            for i in range(n):
                images.append(batch['anchor_image'][i])
                captions.append(f'anchor: {batch["anchor_name"][i]}')

                images.append(batch['neighbor_image'][i])
                captions.append(f'neighbor: {batch["neighbor_name"][i]}')

            # resizing all images to the same size for correct wandb behavior
            resize_image = transforms.Resize((images[0].shape[1], images[0].shape[2]), antialias=True)
            images = [resize_image(i) for i in images]

            # logging the images
            wandb_logger.log_image(
                key=f'{step}_image_sample',
                images=images,
                caption=captions)

    # training
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=1,
        accelerator=get_the_best_accelerator(),
        devices=1,
        callbacks=[
            # TODO: add validation set
            # ModelCheckpoint(mode='min', monitor='val_loss_ssl',
            #                 save_top_k=3, filename='{epoch}-{step}-{val_loss_ssl:.2f}'),
            ModelCheckpoint(every_n_epochs=50),
            LearningRateMonitor('epoch'),
            LogImageSampleCallback()
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader,
        # TODO: add validation set
        # val_dataloaders=val_data_loader
    )

    # saving the final model
    trainer.save_checkpoint(cfg.training.model_save_path)

    wandb.finish()


if __name__ == '__main__':
    main()
