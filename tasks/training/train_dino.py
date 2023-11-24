
import hydra
import lightning.pytorch as pl
import torch
import numpy as np
import wandb
import torchvision.transforms as transforms
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from src.datasets.datasets import get_dataset_class_by_name
from src.models.dino import DINO
from src.utils.wandb import init_run
from src.utils.utils import get_the_best_accelerator
from src.utils.utils import get_cpu_count
from src.utils.callbacks import SSLModeCollapseCallback

from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.dino_transform import DINOTransform


@hydra.main(version_base=None, config_path='../../conf', config_name='train_dino')
def main(cfg: DictConfig):
    init_run(cfg)

    pl.seed_everything(cfg.training.seed)

    # creating datasets
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class('train')
    val_dataset = dataset_class('val')

    transform = None
    # done
    if cfg.dataset.name == 'matek':
        transform = DINOTransform(
            global_crop_size=cfg.dataset.input_size,
            n_local_views=0,
            sigmas=get_sigmas(cfg.dataset.input_size),
            solarization_prob=0.0,
            cj_strength=0.3,
        )
    elif cfg.dataset.name == 'isic':
        transform = DINOTransform(
            global_crop_size=cfg.dataset.input_size,
            n_local_views=0,
            sigmas=get_sigmas(cfg.dataset.input_size),
        )
    elif cfg.dataset.name == 'retinopathy':
        transform = DINOTransform(
            global_crop_size=cfg.dataset.input_size,
            n_local_views=0,
            sigmas=get_sigmas(cfg.dataset.input_size),
        )
    # done
    elif cfg.dataset.name in ('jurkat', 'cifar10'):
        transform = DINOTransform(
            global_crop_size=cfg.dataset.input_size,
            gaussian_blur=(0, 0, 0),
            n_local_views=0,
        )
    else:
        raise ValueError(f'unsupported dataset: {cfg.dataset.name}')

    train_dataset_lightly = LightlyDataset(
        input_dir=train_dataset.images_dir,
        transform=transform
    )

    val_dataset_lightly = LightlyDataset(
        input_dir=val_dataset.images_dir,
        transform=transform
    )

    # augmentations for simclr
    # collate_fn = MultiViewCollate()
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset_lightly,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        # collate_fn=collate_fn,
        drop_last=True,
        num_workers=get_cpu_count()
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset_lightly,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        # collate_fn=collate_fn,
        drop_last=False,
        num_workers=get_cpu_count()
    )

    # defining the model
    lightning_model = DINO(cfg=cfg)

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
            # only first set of images in the batch
            image_name = batch[2][0]

            # getting all views for the first image in the batch
            images = [view[0] for view in batch[0]]

            # creating captions with the image dimensions included
            captions = [
                f'{image_name}\nview {images[i].shape[1]}x{images[i].shape[2]}'
                for i in range(len(images))
            ]

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
            ModelCheckpoint(mode='min', monitor='val_loss_ssl',
                            save_top_k=3, filename='{epoch}-{step}-{val_loss_ssl:.2f}'),
            ModelCheckpoint(every_n_epochs=50, save_top_k=-1),
            LearningRateMonitor('epoch'),
            LogImageSampleCallback(),
            SSLModeCollapseCallback()
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

    wandb.finish()

def round_to_even(number):
    rounded = np.round(number)
    return rounded - (rounded % 2)

def get_local_crop_size(input_size):
    # from the original paper
    coef = 96 / 224
    return int(round_to_even(coef * input_size))

def get_sigmas(input_size):
    old_sigmas = (0.1, 2)
    coef = input_size / 224

    return (old_sigmas[0] * coef, old_sigmas[1] * coef)

if __name__ == '__main__':
    main()
