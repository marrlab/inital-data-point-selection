

import re
import os
import csv
import sys
import numpy as np
import glob
import wandb
import umap
import umap.plot
import torch
import zipfile
import matplotlib.pyplot as plt
import logging
import hydra
import torchvision
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from src.models.lightning_modules import SimCLRModel
from src.datasets.datasets import get_dataset_class_by_name
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from src.utils.utils import load_dataframes
from scipy.stats import entropy
from src.utils.types import Result
from src.utils.utils import load_yaml_as_obj
from src.utils.utils import load_yaml_as_obj, latex_to_pdf, recursive_dict_compare
from src.datasets.datasets import get_dataset_class_by_name
from src.utils.wandb import get_runs
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path='../../conf', config_name='umap_features')
def main(cfg: DictConfig):
    # loading dataset
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    dataset = dataset_class('train', load_images=False,
                            features_path=cfg.features.path)

    if cfg.features.scaling == 'standard':
        dataset.standard_scale_features()

    # getting features to numpy
    xs, ys, zs = [], [], []
    for i in range(len(dataset.features)):
        xs.append(dataset.features[dataset.images_data['names'][i]])
        ys.append(dataset.images_data['labels'][i])
        zs.append(dataset.images_data['labels_text'][i])

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    def run_umap(n_neighbors, min_dist):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        mapper = reducer.fit(xs)

        image_name_prefix = f'n_neighbors={n_neighbors},min_dist={min_dist}'

        umap.plot.points(mapper, cmap='Greys')
        plt.savefig(f'{image_name_prefix}_grey.png', bbox_inches='tight')

        umap.plot.points(mapper, labels=zs, background='white')
        plt.savefig(f'{image_name_prefix}_labels.png', bbox_inches='tight')

    # creating umap images
    for n_neighbors in tqdm(cfg.n_neighbors_options, desc='n_neighbors'):
        for min_dist in tqdm(cfg.min_dist_options, desc='min_dist_options', leave=False):
            run_umap(n_neighbors, min_dist)

    # creating a zip with all the images
    with zipfile.ZipFile('images.zip', mode='w') as zipf:
        for file in glob.glob('*.png'):
            zipf.write(file, arcname=file.split('/')[-1])


if __name__ == '__main__':
    main()