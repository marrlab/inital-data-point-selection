
import torch
import copy
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from src.datasets.datasets import ImageDataset
from typing import List, Tuple, Union
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from src.utils.utils import pick_samples_from_classes_evenly


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


def get_n_kmeans(dataset: ImageDataset, n_samples: int, n_clusters: int, mode='kmeans++', criterium='closest') -> ImageDataset:
    assert dataset.features_path is not None
    assert n_samples <= len(dataset)
    assert n_clusters <= n_samples
    assert mode in ('kmeans++', 'kmeans')
    assert criterium in ('closest', 'furthest', 'random')

    # getting the features
    features = []
    for name in dataset.images_data['names']:
        features.append(dataset.features[name])
    features = np.array(features)

    # getting the cluster centers and feature-to-cluster allegiance
    cluster_centers, cluster_indices = None, None
    if mode == 'kmeans++':
        cluster_centers, _ = kmeans_plusplus(
            features, n_clusters=n_clusters)
    elif mode == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(features)
        cluster_centers = kmeans.cluster_centers_
    cluster_indices, _ = pairwise_distances_argmin_min(features, cluster_centers)
    
    # getting sizes of clusters
    cluster_size_counts = Counter(cluster_indices)
    cluster_sizes = [cluster_size_counts[i] for i in range(n_clusters)]

    # getting the number of points to pick from each cluster
    cluster_samples = pick_samples_from_classes_evenly(cluster_sizes, n_samples)

    indices = []
    for i in range(n_clusters):
        new_indices = None
        distances = np.linalg.norm(features - cluster_centers[i], axis=1)

        if criterium == 'closest':
            distances[cluster_indices != i] = np.inf
            new_indices = np.argsort(distances)[:cluster_samples[i]]
        elif criterium == 'furthest':
            distances[cluster_indices != i] = -np.inf
            new_indices = np.argsort(distances)[::-1][:cluster_samples[i]]
        elif criterium == 'random':
            new_indices = np.random.choice(
                np.where(cluster_indices == i)[0],
                size=cluster_samples[i],
                replace=False)

        indices.extend(new_indices)

    new_dataset = get_by_indices(dataset, indices)

    return new_dataset


