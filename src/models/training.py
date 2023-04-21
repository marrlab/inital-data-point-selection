
import os
import shutil
import torch
import torchvision
import lightning.pytorch as pl
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models.lightning_modules import ImageClassifierLightningModule, SimCLRModel
from src.utils.utils import get_the_best_accelerator
from src.utils.utils import flatten_tensor_dicts
import wandb
from src.datasets.datasets import ImageDataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig


def train_image_classifier(model: torch.nn.Module, train_dataset: ImageDataset, val_dataset: ImageDataset, test_dataset: ImageDataset, cfg: DictConfig):
    assert len(train_dataset.labels) == len(
        val_dataset.labels) == len(test_dataset.labels)
    assert len(train_dataset.classes) == len(
        val_dataset.classes) == len(test_dataset.classes)

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
    test_data_loader = DataLoader(
        test_dataset, batch_size=cfg.training.batch_size)

    wandb_logger = pl.loggers.WandbLogger()

    class LogPredictionSamplesCallback(pl.Callback):
        def on_train_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if trainer.current_epoch % 20 == 0 and batch_idx == 0:
                self._common_batch_end('train', outputs, batch)

        def on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if batch_idx == 0:
                self._common_batch_end('val', outputs, batch)

        def on_test_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self._common_batch_end('test', outputs, batch)

        def _common_batch_end(
                self, step, outputs, batch):
            images = list(batch['image'])
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}'
                        for y_i, y_pred in zip(batch['label'], outputs['pred_labels'])]

            wandb_logger.log_image(
                key=f'{step}_sample_images',
                images=images,
                caption=captions)

    class LogConfusionMatrixCallback(pl.Callback):
        def on_train_epoch_end(
                self, trainer, pl_module):
            self._common_epoch_end('train', pl_module)

        def on_validation_epoch_end(
                self, trainer, pl_module):
            self._common_epoch_end('val', pl_module)

        def on_test_epoch_end(
                self, trainer, pl_module):
            self._common_epoch_end('test', pl_module)

        def _common_epoch_end(
                self, step, pl_module):
            outputs = pl_module.step_outputs[step]
            outputs = flatten_tensor_dicts(outputs)

            wandb.log({
                f'{step}_conf_mat': wandb.plot.confusion_matrix(
                    probs=None,
                    title=f'{step}_conf_mat',
                    y_true=outputs['labels'].cpu().numpy(),
                    preds=outputs['pred_labels'].cpu().numpy(),
                    class_names=train_dataset.labels_text
                ),
            })

            # TODO:
            # wandb.log({
            #     f'{step}_roc': wandb.plot.roc_curve(
            #         ground_truth,
            #         predictions,
            #         labels=None,
            #         classes_to_plot=None
            #     ),
            # })

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
        # check_val_every_n_epoch=20,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator=get_the_best_accelerator(),
        devices=1,
        callbacks=[
            LogPredictionSamplesCallback(),
            LogConfusionMatrixCallback(),
            ModelCheckpoint(mode='max', monitor='train_f1_macro_epoch_end'),
            LearningRateMonitor('epoch'),
        ]
    )

    # the actual training
    trainer.fit(
        lightning_model,
        train_dataloaders=train_data_loader,
        # val_dataloaders=val_data_loader,
    )

    # testing
    trainer.test(ckpt_path='best', dataloaders=test_data_loader)

    # cleaning
    shutil.rmtree('lightning_logs')

    # trainer.save_checkpoint(cfg.training.model_save_path)
