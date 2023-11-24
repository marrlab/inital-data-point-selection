
import numpy as np
from collections import Counter
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from src.utils.utils import pick_samples_from_classes_evenly


def get_n_clustered(features: np.array, n_samples: int, n_clusters: int, mode='kmeans++', criterium='closest', verbose=False):
    assert n_clusters <= n_samples
    assert mode in ('kmeans++', 'kmeans')
    assert criterium in ('closest', 'furthest', 'random', 'half_in_half', 'fps')

    # getting the cluster centers and feature-to-cluster allegiance
    cluster_centers, cluster_indices = None, None
    if mode == 'kmeans++':
        cluster_centers, _ = kmeans_plusplus(
            features, n_clusters=n_clusters)
    elif mode == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto').fit(features)
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
        elif criterium == 'half_in_half':
            new_indices = []
            closest_cluster_samples = int(np.ceil(cluster_samples[i] / 2))
            furthest_cluster_samples = cluster_samples[i] - closest_cluster_samples

            # half from closest
            distances[cluster_indices != i] = np.inf
            new_indices.extend(np.argsort(distances)[:closest_cluster_samples])

            distances[cluster_indices != i] = -np.inf
            new_indices.extend(np.argsort(distances)[::-1][:furthest_cluster_samples])
        elif criterium == 'fps':
            cluster_features = features[cluster_indices == i]

            # select the first point randomly
            new_indices = [np.random.randint(cluster_features.shape[0])]

            # select the first point closest to the cluster center
            # new_indices = [np.argmin(np.linalg.norm(cluster_features - cluster_centers[i], axis=1))]

            dists = np.linalg.norm(cluster_features - cluster_features[new_indices[0]], axis=1)

            for _ in range(1, cluster_samples[i]):
                # select the point with the largest distance to the nearest indices point
                new_indices.append(np.argmax(dists))
                dists = np.minimum(dists, np.linalg.norm(cluster_features - cluster_features[new_indices[-1]], axis=1))

            # reindexing back
            new_indices = np.where(cluster_indices == i)[0][new_indices]

        indices.extend(new_indices)

    if verbose:
        return indices, cluster_indices, cluster_centers
    
    return indices

