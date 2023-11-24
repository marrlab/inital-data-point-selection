
import torch
import copy
import random
import numpy as np
from src.datasets.datasets import ImageDataset
from typing import List, Tuple, Union
from src.sampling.clustering import get_n_clustered as get_n_clustered_features


def get_by_indices(dataset: ImageDataset, indices: List[int]) -> ImageDataset:
    new_dataset = copy.deepcopy(dataset)

    new_dataset.images_data['names'] = np.array(new_dataset.images_data['names'])[indices].tolist()
    new_dataset.images_data['labels_text'] = np.array(new_dataset.images_data['labels_text'])[indices].tolist()
    new_dataset.images_data['labels'] = np.array(new_dataset.images_data['labels'])[indices].tolist()
    new_dataset.images_data['paths'] = np.array(new_dataset.images_data['paths'])[indices].tolist()

    return new_dataset


def get_by_names(dataset: ImageDataset, names: List[str]) -> ImageDataset:
    indices = []
    for i, name in enumerate(dataset.images_data['names']):
        if name in names:
            indices.append(i)

    new_dataset = get_by_indices(dataset, indices)

    return new_dataset


def get_n_random(dataset: ImageDataset, n: int) -> ImageDataset:
    assert n <= len(dataset)

    indices = random.sample(range(len(dataset)), k=n)
    new_dataset = get_by_indices(dataset, indices)

    return new_dataset


def get_n_sorted_by_feature_func(dataset: ImageDataset, n: int, func, n_smallest=True) -> ImageDataset:
    assert dataset.features_path is not None
    assert n <= len(dataset)

    feature_func_values = []
    for data_point in dataset:
        feature_func_values.append(func(data_point['feature']))

    indices = np.argsort(feature_func_values)[:n]
    new_dataset = get_by_indices(dataset, indices)

    return new_dataset


def get_n_clustered(dataset: ImageDataset, n_samples: int, n_clusters: int, mode='kmeans++', criterium='closest') -> ImageDataset:
    assert dataset.features_path is not None
    assert n_samples <= len(dataset)

    # getting the features
    features = []
    for name in dataset.images_data['names']:
        features.append(dataset.features[name])
    features = np.array(features)

    indices = get_n_clustered_features(features, n_samples, n_clusters, mode, criterium)

    new_dataset = get_by_indices(dataset, indices)

    return new_dataset
