

import re
import os
import csv
import sys
import wandb
import torch
import logging
import hydra
import torchvision
import pandas as pd
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

@hydra.main(version_base=None, config_path='../../conf', config_name='classification_metrics')
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

    # random baseline
    def filter_run_random(row):
        return (
            recursive_dict_compare(row['config']['dataset'], cfg_dict['dataset']) and
            recursive_dict_compare(row['config']['training'], cfg_dict['training']) and
            row['summary']['epoch'] == (cfg_dict['training']['epochs'] - 1) and
            all(rm in row['summary'] for rm in required_metrics)
        )

    def transform_runs_random(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)

        df['CLASSES'] = df.apply(lambda r: r['config']['classes'], axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df_random = get_runs('random-baseline')
    filtered_rows = df_random.apply(filter_run_random, axis=1)
    df_random = df_random[filtered_rows]
    df_random = transform_runs_random(df_random)
    df_random['method'] = 'random'

    # badge sampling
    def filter_run_badge(row):
        return (
            filter_run_random(row) and
            recursive_dict_compare(row['config']['features'], cfg_dict['features'])
        )

    def transform_runs_badge(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)

        df['CLASSES'] = df.apply(lambda r: r['config']['classes'], axis=1)
        df['method'] = df.apply(
            lambda r: f"{r['config']['kmeans']['mode']} {r['config']['kmeans']['criterium']}", axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df_badge = get_runs('badge-sampling')
    filtered_rows = df_badge.apply(filter_run_badge, axis=1)
    df_badge = df_badge[filtered_rows]
    df_badge = transform_runs_badge(df_badge)

    # joining
    df = pd.concat([df_random, df_badge], ignore_index=True)

    # computing statistics
    df_mean = df.groupby(['method']).mean()
    df_mean = df_mean.sort_values(by='F1 MACRO', ascending=False)
    df_mean_style = df_mean.style.highlight_max(
        props='font-weight:bold;').format(precision=2)

    df_std = df.groupby(['method']).std()
    df_std = df_std.loc[df_mean.index]
    df_std_style = df_std.style.highlight_min(
        props='font-weight:bold;').format(precision=2)

    # combining statistics
    def round_and_convert(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns.to_list():
            df[c] = df[c].map('{:.2f}'.format)

        return df

    df_mean_std = round_and_convert(
        df_mean) + u'\u00B1' + round_and_convert(df_std)
    df_mean_std_style = df_mean_std.style.highlight_max(
        props='font-weight:bold;').format(precision=2)

    # exporting
    df_mean_style.to_latex(buf='df_mean.tex', hrules=True, convert_css=True)
    latex_to_pdf('df_mean.tex', 'df_mean.pdf')
    df_std_style.to_latex('df_std.tex', hrules=True, convert_css=True)
    latex_to_pdf('df_std.tex', 'df_std.pdf')
    df_mean_std_style.to_latex('df_mean_std.tex', hrules=True, convert_css=True)
    latex_to_pdf('df_mean_std.tex', 'df_mean_std.pdf')

if __name__ == '__main__':
    main()