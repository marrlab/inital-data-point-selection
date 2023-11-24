
import os
import shutil
import torch
import torchvision
import lightning.pytorch as pl
from typing import Union, Tuple, Iterable
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models.lightning_modules import ImageClassifierLightningModule
from src.models.simclr import SimCLR
from src.utils.utils import get_the_best_accelerator
from src.utils.utils import flatten_tensor_dicts
import wandb
from src.datasets.datasets import ImageDataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig
from src.utils.utils import get_cpu_count


def train_image_classifier(
        model: torch.nn.Module, 
        train_dataset: ImageDataset, 
        val_dataset: Union[ImageDataset, None], 
        test_dataset: Union[ImageDataset, None], 
        cfg: DictConfig,
        log_images: bool = True):
    
    # asserts
    assert train_dataset is not None or test_dataset is not None

    if train_dataset is not None and val_dataset is not None:
        assert len(train_dataset.labels) == len(val_dataset.labels)
        assert len(train_dataset.classes) == len(val_dataset.classes)
    if train_dataset is not None and test_dataset is not None:
        assert len(train_dataset.labels) == len(test_dataset.labels)
        assert len(train_dataset.classes) == len(test_dataset.classes)

    # setting up oversampling if chosen
    train_data_loader = None
    if train_dataset is not None:
        if cfg.training.oversample:
            class_counts = defaultdict(int)
            for i in range(len(train_dataset)):
                class_counts[train_dataset[i]['class']] += 1

            weights = [1 / class_counts[train_dataset[i]['class']]
                    for i in range(len(train_dataset))]

            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=max(cfg.training.batch_size, len(train_dataset)),
                replacement=True
            )
            train_data_loader = DataLoader(
                train_dataset, batch_size=cfg.training.batch_size, sampler=sampler, 
                num_workers=get_cpu_count())
        else:
            train_data_loader = DataLoader(
                train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                num_workers=get_cpu_count())

    # validation
    val_data_loader = None
    if val_dataset is not None:
        val_data_loader = DataLoader(
            val_dataset, batch_size=cfg.training.batch_size, shuffle=True, 
            num_workers=get_cpu_count())

    # testing
    test_data_loader = None
    if test_dataset is not None:
        test_data_loader = DataLoader(
            test_dataset, batch_size=cfg.training.batch_size, shuffle=True, 
            num_workers=get_cpu_count())

    wandb_logger = pl.loggers.WandbLogger()

    # for the purposes of determing the properties of a dataset
    main_dataset = None
    if train_dataset is not None:
        main_dataset = train_dataset
    else:
        main_dataset = test_dataset

    class LogPredictionSamplesCallback(pl.Callback):
        def on_train_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if trainer.current_epoch % 100 == 0 and batch_idx == 0:
                self._common_batch_end('train', outputs, batch)

        def on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            pass

        def on_test_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if batch_idx == 0:
                self._common_batch_end('test', outputs, batch)

        def _common_batch_end(
                self, step, outputs, batch):
            n = 32
            images = list(batch['image'][:n])
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}'
                        for y_i, y_pred in zip(batch['label'][:n], outputs['pred_labels'][:n])]

            wandb_logger.log_image(
                key=f'{step}_sample_images',
                images=images,
                caption=captions)

    class LogConfusionMatrixCallback(pl.Callback):
        def on_train_epoch_end(
                self, trainer, pl_module):
            pass

        def on_validation_epoch_end(
                self, trainer, pl_module):
            pass

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
                    class_names=main_dataset.labels_text
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
        main_dataset.get_number_of_labels(),
        main_dataset.get_number_of_classes(),
        main_dataset.label_to_class_mapping.copy(),
        main_dataset.class_to_label_mapping.copy(),
        cfg
    )

    wandb_logger.watch(lightning_model)

    # initializing callbacks
    callbacks = [
        ModelCheckpoint(mode='max', monitor='train_f1_macro_epoch_end'),
        LearningRateMonitor('epoch'),
        LogConfusionMatrixCallback(),
    ]
    if log_images:
        callbacks.append(LogPredictionSamplesCallback())

    # training
    ret = None
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        # check_val_every_n_epoch=20,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator=get_the_best_accelerator(),
        devices=1,
        callbacks=callbacks
    )

    # the actual training
    if train_data_loader is not None:
        trainer.fit(
            lightning_model,
            train_dataloaders=train_data_loader,
        )

    # validation
    if val_data_loader is not None:
        ret = trainer.validate(lightning_model, dataloaders=val_data_loader)

    # testing
    if test_data_loader is not None:
        ret = trainer.test(lightning_model, dataloaders=test_data_loader)

    # cleaning
    logs_dir = 'lightning_logs'
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

    # saving final model
    if cfg.training.model_save_path is not None:
        print(f'saving final model to {cfg.training.model_save_path}')
        trainer.save_checkpoint(cfg.training.model_save_path)

    return ret
