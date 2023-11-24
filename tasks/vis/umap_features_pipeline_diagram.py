
import os
import random
import numpy as np
import glob
import umap
import umap.plot
import zipfile
import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from src.datasets.datasets import get_dataset_class_by_name
from src.datasets.datasets import get_dataset_class_by_name
from src.models.build_data_path import get_vis_folder_path, get_features_path, get_scan_features_path
from src.sampling.clustering import get_n_clustered
import warnings
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path='../../conf', config_name='umap_features_pipeline_diagram')
def main(cfg: DictConfig):
    assert cfg.n_neighbors is not None
    assert cfg.min_dist is not None

    print(f'saving everything to: {os.getcwd()}')

    # loading dataset
    features_path = None
    if cfg.use_scan_weights:
        features_path = get_scan_features_path(cfg)
    else:
        features_path = get_features_path(cfg)

    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    dataset = dataset_class('train', load_images=False,
                            features_path=features_path)

    if cfg.features.scaling == 'standard':
        dataset.standard_scale_features()

    # getting features to numpy
    xs, ys, zs = [], [], []
    for i in range(len(dataset.features)):
        xs.append(dataset.features[dataset.images_data['names'][i]])
        ys.append(dataset.images_data['labels'][i])
        zs.append(dataset.images_data['labels_text'][i])

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # runing umap
    reducer = umap.UMAP(n_neighbors=cfg.n_neighbors, min_dist=cfg.min_dist)
    features = reducer.fit_transform(xs)

    # saving features
    np.save('features', features)

    # filtering features for better vizualization
    indices_filtered = random.sample(list(range(features.shape[0])), cfg.n_filtered)
    features_filtered = features[indices_filtered, :]

    # sampling features
    indices_sampled = get_n_clustered(
        features=features_filtered, 
        n_samples=cfg.n_samples, 
        n_clusters=1, 
        mode='kmeans', 
        criterium='fps'
    )
    features_sampled = features_filtered[indices_sampled, :]

    # creating the images
    # filtered
    plt.axis('off')

    plt.scatter(
        features_filtered[:, 0], features_filtered[:, 1], 
        color='#bababa', marker='.', s=10,
    )

    plt.savefig('filtered.pdf')
    plt.clf()

    # sampled
    plt.axis('off')

    plt.scatter(
        features_filtered[:, 0], features_filtered[:, 1], 
        color='#bababa', marker='.', s=10,
    )
    plt.scatter(
        features_sampled[:, 0], features_sampled[:, 1], 
        color='black', marker='o', s=30,
    )
    plt.scatter(
        features_sampled[:, 0], features_sampled[:, 1], 
        color='white', marker='o', s=12,
    )

    plt.savefig('sampled.pdf')
    plt.clf()

    # annotated
    plt.axis('off')

    plt.scatter(
        features_filtered[:, 0], features_filtered[:, 1], 
        color='#bababa', marker='.', s=10,
    )
    plt.scatter(
        features_sampled[:, 0], features_sampled[:, 1], 
        color='black', marker='o', s=30,
    )

    plt.savefig('annotated.pdf')
    plt.clf()



if __name__ == '__main__':
    main()