
import os
import uuid
import time
import hydra
import torch
import copy
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from collections import defaultdict
import torchvision
import lightning.pytorch as pl
from src.datasets.datasets import get_dataset_class_by_name, ImageFeaturesDataset
from src.models.classifiers import get_ssl_preprocess, get_ssl_transform, get_classifier_from_ssl
from src.models.scan import get_classifier_from_scan
from src.datasets.subsets import get_n_clustered
from src.models.training import train_image_classifier
from src.models.build_data_path import get_features_path, get_scan_features_path, get_precomputed_features_folder_path
from src.utils.wandb import init_run, cast_dict_to_int

@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_ceiling')
def main(cfg: DictConfig):
    # # debug
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return

    assert cfg.use_scan_weights == False
    assert cfg.training.learning_rate is None
    assert cfg.training.train_samples is None

    # we don't want this due to different sampling each time
    if cfg.training.seed is not None:
        print(f'setting random seed: {cfg.training.seed}')
        pl.seed_everything(cfg.training.seed)

    # loading reatures
    features_path = None
    if cfg.use_scan_weights:
        features_path = get_scan_features_path(cfg, absolute=False)
    else:
        features_path = get_features_path(cfg, absolute=False)
    print(f'using features stored at the path: {features_path}')

    # loading datasets
    preprocess = get_ssl_preprocess(cfg)
    transform = get_ssl_transform(cfg)    

    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class(
        'train', 
        features_path=features_path,
        transform=transform, 
        preprocess=torchvision.transforms.Resize(cfg.dataset.input_size, antialias=True)
    )
    val_dataset = dataset_class('val', preprocess=preprocess)
    test_dataset = dataset_class('test', preprocess=preprocess)

    # hyperparameter search
    models = []
    metrics = []

    feature_dim = train_dataset.get_feature_dim()
    num_classes = train_dataset.get_number_of_classes()

    print(f'training for these learning rates: {cfg.learning_rates}')
    for learning_rate in cfg.learning_rates:
        print(f'training a linear head: learning_rate={learning_rate}')

        # initializing the model
        model = get_classifier_from_ssl(cfg, feature_dim, num_classes)
    
        cfg.training.learning_rate = learning_rate

        init_run(cfg)

        start = time.time()
        m = train_image_classifier(
            model=model, 
            train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            test_dataset=None,
            cfg=cfg,
        )[0]
        wandb.config.training_time = time.time() - start

        wandb.finish()

        # saving
        models.append(model)
        metrics.append(m)

    # determining the best model
    val_f1_macros = [m['val_f1_macro_epoch_end'] for m in metrics]
    print(f'val_f1_macro_epoch_end: {val_f1_macros}')
    best_model = models[np.argmax(val_f1_macros)]

    cfg.wandb.project = f'{cfg.wandb.project}-best'

    init_run(cfg)

    train_image_classifier(
        model=best_model, 
        train_dataset=None, 
        val_dataset=None, 
        test_dataset=test_dataset,
        cfg=cfg,
    )

    wandb.finish()

if __name__ == '__main__':
    main()
