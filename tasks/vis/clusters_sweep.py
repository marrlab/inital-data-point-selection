

import re
import os
import csv
import sys
import wandb
import torch
import logging
import hydra
import torchvision
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


@hydra.main(version_base=None, config_path='../../conf', config_name='clusters_sweep')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')

    # defined fields
    required_metrics = [
        'val_accuracy_epoch_end',
        'val_f1_macro_epoch_end',
        'val_balanced_accuracy_epoch_end',
        'val_matthews_corrcoef_epoch_end',
        'val_cohen_kappa_score_epoch_end'
    ]
    required_metrics_name = [
        'ACC',
        'F1 MACRO',
        'BA',
        'MCC',
        'CKS'
    ]

    # general
    def filter_run_general(row):
        try:
            return (
                recursive_dict_compare(row['config']['dataset'], cfg_dict['dataset']) and
                recursive_dict_compare(row['config']['training'], cfg_dict['training']) and
                recursive_dict_compare(row['config']['features'], cfg_dict['features']) and
                row['summary']['epoch'] == (cfg_dict['training']['epochs'] - 1) and
                all(rm in row['summary'] for rm in required_metrics)
            )
        except KeyError:
            return False

    def transform_runs_general(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)

        df['clusters'] = df.apply(
            lambda r: r['config']['kmeans']['clusters'], axis=1)
        df['criterium'] = df.apply(
            lambda r: r['config']['kmeans']['criterium'], axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    # random baseline

    def filter_run_random(row):
        try:
            return (
                filter_run_general(row) and
                row['config']['kmeans']['clusters'] == 1
            )
        except KeyError:
            return False

    df_random = get_runs('badge-sampling')
    filtered_rows = df_random.apply(filter_run_random, axis=1)
    df_random = df_random[filtered_rows]
    df_random = transform_runs_general(df_random)

    # badge sampling
    def filter_run_badge(row):
        try:
            return (
                filter_run_general(row) and
                row['config']['kmeans']['clusters'] > 1 and
                row['config']['kmeans']['mode'] == 'kmeans'
            )
        except KeyError:
            return False

    df_badge = get_runs('badge-sampling')
    filtered_rows = df_badge.apply(filter_run_badge, axis=1)
    df_badge = df_badge[filtered_rows]
    df_badge = transform_runs_general(df_badge)

    # saving the dataframes
    df_random.to_pickle('df_random.pickle')
    df_badge.to_pickle('df_badge.pickle')

    # creating the plots
    for metric_name in required_metrics_name:
        random_mean = df_random.mean()[metric_name]
        random_std = df_random.std()[metric_name]

        fig, ax = plt.subplots()
        ax.set_xticks(df_badge['clusters'].unique())
        ax.axvline(cfg.clusters_highlight, color='black',
                   linestyle='--', label='number of classes')
        sns.lineplot(data=df_badge, x='clusters', y=metric_name, hue='criterium', hue_order=[
                     'random', 'closest', 'furthest'], errorbar='sd', marker='o', ax=ax)
        ax.axhline(random_mean, color='black', label='random baseline')
        ax.axhspan(random_mean - random_std, random_mean +
                   random_std, color='black', alpha=0.1)
        ax.legend()

        fig.savefig(f'{metric_name}.pdf')


if __name__ == '__main__':
    main()
