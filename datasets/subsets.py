
import torch
import copy
import random
import numpy as np
from datasets.datasets import ImageDataset
from typing import List

def get_by_indices(dataset: ImageDataset, indices: List[int]) -> ImageDataset:
    new_dataset = copy.deepcopy(dataset)

    new_dataset.images_data['names'] = list(np.array(new_dataset.images_data['names'])[indices])
    new_dataset.images_data['labels_text'] = list(np.array(new_dataset.images_data['labels_text'])[indices])
    new_dataset.images_data['labels'] = list(np.array(new_dataset.images_data['labels'])[indices])
    new_dataset.images_data['paths'] = list(np.array(new_dataset.images_data['paths'])[indices])

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
    for data_point in enumerate(dataset):
        feature_func_values.append(func(data_point['feature']))
        
    indices = np.argsort(feature_func_values)[:n]
    new_dataset = get_by_indices(dataset, indices)

    return new_dataset
