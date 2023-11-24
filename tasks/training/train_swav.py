
import hydra
import lightning.pytorch as pl
import torch
import wandb
import numpy as np
import torchvision.transforms as transforms
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightly.transforms.swav_transform import SwaVTransform
from omegaconf import DictConfig

from src.datasets.datasets import get_dataset_class_by_name
from src.models.swav import SwaV
from src.utils.wandb import init_run
from src.utils.utils import get_the_best_accelerator
from src.utils.utils import get_cpu_count


# TODO: try with other self-supervised models
@hydra.main(version_base=None, config_path='../../conf', config_name='train_swav')
def main(cfg: DictConfig):
    init_run(cfg)

    pl.seed_everything(cfg.training.seed)

    # creating datasets
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class('train')
    val_dataset = dataset_class('val')

    # some augmentations
    transform = SwaVTransform(
        crop_sizes=(cfg.dataset.input_size, get_smaller_crop_size(cfg.dataset.input_size)),

        # turns off multi-crop
        crop_counts=(2, 0),
        # second element in tuples is not used, but needs to be specified nonetheless
        crop_min_scales=(0.25, 0.05),
        crop_max_scales=(1.00, 0.14),

        # tinkering with other augmentations (apparently not needed for cifar10 and isic)
        cj_strength=0.2,
    )

    train_dataset_lightly = LightlyDataset(
        input_dir=train_dataset.images_dir,
        transform=transform
    )

    val_dataset_lightly = LightlyDataset(
        input_dir=val_dataset.images_dir,
        transform=transform
    )

    collate_fn = MultiViewCollate()

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset_lightly,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=get_cpu_count()
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset_lightly,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=get_cpu_count()
    )

    # defining the model
    lightning_model = SwaV(cfg=cfg)

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
        val_dataloaders=val_data_loader
    )

    # saving the final model
    trainer.save_checkpoint(cfg.training.model_save_path)

    wandb.finish()


def round_to_even(number):
    rounded = np.round(number)
    return rounded - (rounded % 2)

def get_smaller_crop_size(input_size):
    # from the original paper
    coef = 96 / 224
    return int(round_to_even(coef * input_size))


if __name__ == '__main__':
    main()
