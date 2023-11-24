
import os
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
import warnings
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path='../../conf', config_name='umap_features')
def main(cfg: DictConfig):
    folder_path = get_vis_folder_path(cfg)
    print(f'saving everything to: {folder_path}')

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

    def run_umap(n_neighbors, min_dist):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        mapper = reducer.fit(xs)

        image_name_prefix = f'n_neighbors={n_neighbors},min_dist={min_dist}'

        umap.plot.points(mapper, cmap='Greys')
        plt.savefig(f'{image_name_prefix}_grey.png', bbox_inches='tight')

        umap.plot.points(mapper, labels=zs, background='white')
        plt.savefig(os.path.join(folder_path, f'{image_name_prefix}_labels.png'), bbox_inches='tight')

    # creating umap images
    for n_neighbors in tqdm(cfg.n_neighbors_options, desc='n_neighbors'):
        for min_dist in tqdm(cfg.min_dist_options, desc='min_dist_options', leave=False):
            run_umap(n_neighbors, min_dist)

    # creating a zip with all the images
    with zipfile.ZipFile('images.zip', mode='w') as zipf:
        for file in glob.glob('*.png'):
            zipf.write(file, arcname=file.split('/')[-1])


if __name__ == '__main__':
    main()