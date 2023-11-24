
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
from src.vis.helpers import format_ssl
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['legend.loc'] = 'upper left'
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 20})

@hydra.main(version_base=None, config_path='../../conf', config_name='umap_features_appendix')
def main(cfg: DictConfig):
    assert cfg.n_neighbors is not None
    assert cfg.min_dist is not None

    # plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['legend.loc'] = cfg.legend_loc

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
    mapper = reducer.fit(xs)

    fig = umap.plot.points(mapper, labels=zs, theme='blue', show_legend=cfg.show_legend)
    for t in fig.texts:
        t.set_visible(False)

    if cfg.set_title:
        fig.set_title(format_ssl(cfg.training.weights.type))
    
    plt.savefig(f'{cfg.dataset.name}_{cfg.training.weights.type}_{cfg.training.weights.version}.pdf', bbox_inches='tight')
    
if __name__ == '__main__':
    main()