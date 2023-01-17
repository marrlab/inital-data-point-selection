
import torch
import copy
import random
import numpy as np
from datasets.datasets import ImageDataset
from typing import List, Tuple
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def get_by_indices(dataset: ImageDataset, indices: List[int]) -> ImageDataset:
    new_dataset = copy.deepcopy(dataset)

    new_dataset.images_data['names'] = list(
        np.array(new_dataset.images_data['names'])[indices])
    new_dataset.images_data['labels_text'] = list(
        np.array(new_dataset.images_data['labels_text'])[indices])
    new_dataset.images_data['labels'] = list(
        np.array(new_dataset.images_data['labels'])[indices])
    new_dataset.images_data['paths'] = list(
        np.array(new_dataset.images_data['paths'])[indices])

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


def get_n_kmeans(dataset: ImageDataset, n: int, mode='kmeans++', criterium='closest', verbose=False) -> Tuple[ImageDataset, tuple[ImageDataset, np.array, np.array, np.array]]:
    assert dataset.features_path is not None
    assert n <= len(dataset)
    assert mode in ('kmeans++', 'kmeans')
    assert criterium in ('closest', 'furthest')

    features = []
    for name in dataset.images_data['names']:
        features.append(dataset.features[name])

    features = np.array(features)

    centers, indices = None, None
    if mode == 'kmeans++':
        centers, indices = kmeans_plusplus(
            features, n_clusters=n, random_state=0)
    elif mode == 'kmeans':
        kmeans = KMeans(n_clusters=n).fit(features)
        centers = kmeans.cluster_centers_
        indices, _ = pairwise_distances_argmin_min(centers, features)

    if criterium == 'furthest':
        # getting cluster index for each feature
        cluster_indices, cluster_distances = pairwise_distances_argmin_min(
            features, centers)

        indices = []
        # finding a point belonging to cluster i that has the biggest distance to its center
        for i in range(n):
            original_indices = np.where(cluster_indices == i)[0]
            furthest_feature_index = original_indices[np.argmax(
                cluster_distances[original_indices])]

            indices.append(furthest_feature_index)

    new_dataset = get_by_indices(dataset, indices)

    if verbose:
        return new_dataset, centers, indices, features[indices]
    else:
        return new_dataset
