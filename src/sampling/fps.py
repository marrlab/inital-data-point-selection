

import numpy as np
from collections import Counter
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from src.utils.utils import pick_samples_from_classes_evenly


def fps(features: np.array, n_samples: int, metric='l2', verbose=False):
    assert n_samples <= len(features)
    assert metric in ('l1', 'l2', 'inner_product')

    print(f'fps features shape: {features.shape}')
    features = features.copy()
    original_shape = features.shape
    if metric == 'inner_product':
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        assert np.abs(np.linalg.norm(features[0]) - 1.0) < 1e-7

    assert features.shape == original_shape

    def dist_to_features(a):
        if metric == 'l1':
            return np.sum(np.abs(features - a), axis=1)
        elif metric == 'l2':
            return np.linalg.norm(features - a, axis=1)
        elif metric == 'inner_product':
            return np.arccos(np.clip(np.dot(features, a), -1, 1))
        else:
            raise ValueError(f'\'{metric}\' is not supported')

    indices = [np.random.randint(len(features))]
    dists = np.inf * np.ones(len(features))
    for _ in range(n_samples - 1):
        dists = np.minimum(dists, dist_to_features(features[indices[-1]]))
        new_index = np.argmax(dists)
        indices.append(new_index)

    assert len(indices) == n_samples

    return indices
