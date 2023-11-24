
import re
import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from src.datasets.datasets import get_dataset_class_by_name
from src.vis.constants import HALF_IN_HALF_TRAIN_SAMPLES_TO_CLUSTERS


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def digits_before_zero(n):
    s = str(n)
    before_zero = re.search(r'^([^.]+)', s).group(1)

    return len(before_zero)

def mean_plus_minus_std(df_mean, df_std, decimal=2):
    # # determining the maximum number of decimal places before zero
    # decimals_max_n = digits_before_zero(df_mean.max().max())
    # decimals_min_n = digits_before_zero(df_mean.min().min())
    # decimals = max(decimals_max_n, decimals_min_n)

    d_mean = df_mean.to_dict()
    d_std = df_std.to_dict()
    d_ret = deepcopy(d_mean)

    format_str = f'{{:.{decimal}f}}'
    for k1 in d_mean:
        # determining the maximum number of decimal places before zero
        decimals_max_n = digits_before_zero(max(df_mean[k1]))
        decimals_min_n = digits_before_zero(min(df_mean[k1]))
        decimals_mean = max(decimals_max_n, decimals_min_n)

        decimals_max_n = digits_before_zero(max(df_std[k1]))
        decimals_min_n = digits_before_zero(min(df_std[k1]))
        decimals_std = max(decimals_max_n, decimals_min_n)

        for k2 in d_mean[k1]:
            mean_str = format_str.format(round(d_mean[k1][k2], decimal))
            decimals_diff = decimals_mean - digits_before_zero(mean_str)
            mean_str = decimals_diff * '\\enspace' + mean_str

            std_str = format_str.format(round(d_std[k1][k2], decimal))
            decimals_diff = decimals_std - digits_before_zero(std_str)
            std_str = decimals_diff * '\\enspace' + std_str

            d_ret[k1][k2] = mean_str + u'\u00B1' + std_str

    df_ret = pd.DataFrame(d_ret)

    return df_ret


def format_criterium(criterium, train_samples=100):
    criterium_formatted = None

    if criterium == 'fps':
        criterium_formatted = 'furthest point sampling'
    elif criterium in ['closest', 'furthest']:
        criterium_formatted = criterium.replace('_', ' ')
        if train_samples is not None:
            criterium_formatted = f'{criterium_formatted} (k={train_samples})'
    elif criterium == 'half_in_half':
        criterium_formatted = 'closest/furthest'
        if train_samples is not None:
            criterium_formatted = f'{criterium_formatted} (k={HALF_IN_HALF_TRAIN_SAMPLES_TO_CLUSTERS[train_samples]})'
    else:
        criterium_formatted = criterium.replace('_', ' ')

    return criterium_formatted


def format_dataset(dataset):
    return {
        'matek': 'Matek',
        'isic': 'ISIC',
        'retinopathy': 'Retinopathy',
        'jurkat': 'Jurkat'
    }[dataset]
 


def format_ssl(ssl):
    ssl_formatted = None
    if 'simclr' in ssl:
        ssl_formatted = 'SimCLR'
    elif 'swav' in ssl:
        ssl_formatted = 'SwAV'
    elif 'dino' in ssl:
        ssl_formatted = 'DINO'
    else:
        raise ValueError(f'unsupported ssl method: {ssl}')

    return ssl_formatted


def get_class_counts(dataset_name):
    # creating the dataset
    dataset_class = get_dataset_class_by_name(dataset_name)

    class_directories = []
    for split in ['train', 'val', 'test']:
        dataset = dataset_class(split=split)

        # searching through image directories
        images_dir = dataset.images_dir
        class_directories.extend(glob.glob(os.path.join(images_dir, '*')))

    # count the number of images in each class
    class_label_to_count = defaultdict(int)
    for class_dir in class_directories:
        class_label = os.path.basename(class_dir)
        image_files = glob.glob(os.path.join(class_dir, '*'))  # Change the extension as per your dataset
        class_count = len(image_files)

        class_label_to_count[class_label] += class_count

    return dict(class_label_to_count)

def get_class_examples(dataset_name):
    # creating the dataset
    dataset_class = get_dataset_class_by_name(dataset_name)
    dataset = dataset_class(split='train')

    labels_text = dataset.images_data['labels_text']
    paths = dataset.images_data['paths']

    label_text_to_paths = defaultdict(list)

    for i in range(len(labels_text)):
        label_text_to_paths[labels_text[i]].append(paths[i])

    label_text_to_example_path = {}
    for label_text in label_text_to_paths:
        label_text_to_example_path[label_text] = \
            random.choice(label_text_to_paths[label_text])

    return label_text_to_example_path

def get_grayscale_color(intensity):
    return (intensity, intensity, intensity)
