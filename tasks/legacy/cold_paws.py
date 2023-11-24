
import numpy as np
import csv
import re
import hydra
import copy
import wandb
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import torchvision
import lightning.pytorch as pl
from src.utils.utils import get_cpu_count, to_best_available_device
from src.datasets.datasets import get_dataset_class_by_name
from src.models.classifiers import get_ssl_preprocess, get_classifier_from_ssl, get_ssl_transform, get_ssl_model_class
from src.models.scan import get_classifier_from_scan
from src.datasets.subsets import get_by_indices
from src.models.training import train_image_classifier
from src.models.build_data_path import get_cold_paws_path, get_model_path
from src.utils.wandb import init_run, cast_dict_to_int
from src.sampling.fps import fps


@hydra.main(version_base=None, config_path='../../conf', config_name='cold_paws')
def main(cfg: DictConfig):
    # # debug
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return

    init_run(cfg)

    # we don't want this due to different sampling each time
    # pl.seed_everything(cfg.training.seed)

    preprocess = get_ssl_preprocess(cfg)
    transform = get_ssl_transform(cfg)    

    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class(
        'train', 
        preprocess=preprocess
    )
    # val_dataset = dataset_class('val', preprocess=preprocess)
    test_dataset = dataset_class('test', preprocess=preprocess)

    # subsetting the train dataset using cold paws list of image names that were sampled
    sampled_names = []
    with open(get_cold_paws_path(cfg), 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            sampled_names.append(row['name'])

    print(f'cold paws sampled names: {sampled_names}')
    print(f'total: {len(sampled_names)}')
    print()

    # getting indices of thoes sampled names
    indices = []
    for i, name in enumerate(train_dataset.images_data['names']):
        if name in sampled_names:
            indices.append(i)

    print(f'cold paws sampled indices: {indices}')
    print(f'total: {len(indices)}')
    print()

    # subset
    train_subset = get_by_indices(train_dataset, indices)
    train_subset.transform = transform
    train_subset.preprocess = torchvision.transforms.Resize(cfg.dataset.input_size, antialias=True)

    train_subset.reassign_classes()
    label_counts = defaultdict(int)
    for i in range(len(train_subset)):
        label_counts[train_subset[i]['label']] += 1
    wandb.config.labels = train_subset.get_number_of_labels()
    wandb.config.labels_text = train_subset.labels_text
    wandb.config.label_counts = cast_dict_to_int(label_counts)
    wandb.config.label_to_class_mapping = cast_dict_to_int(
        train_subset.label_to_class_mapping)
    wandb.config.class_to_label_mapping = cast_dict_to_int(
        train_subset.class_to_label_mapping)
    wandb.config.classes = train_subset.get_number_of_classes()

    # val_dataset.match_classes(train_subset)
    test_dataset.match_classes(train_subset)

    num_classes = train_subset.get_number_of_classes()

    model = None
    if cfg.use_scan_weights:
        raise ValueError('cannot use scan weights')
    else:
        print('using ssl weights')
        model = get_classifier_from_ssl(cfg, 512, num_classes)
    train_image_classifier(model, train_subset, None, test_dataset, cfg)

    wandb.finish()


if __name__ == '__main__':
    main()
