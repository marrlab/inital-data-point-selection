
import glob
import os
import hydra
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
from matplotlib import gridspec
from omegaconf import DictConfig, OmegaConf
from src.datasets.datasets import get_dataset_class_by_name
from src.utils.hydra import get_original_cwd_safe
from src.vis.helpers import get_class_counts

DATASET_NAMES = ('matek', 'isic', 'retinopathy', 'jurkat', 'cifar10')
SAMPLES_PATH = 'src/datasets/data/samples'
TICK_LABEL_FONT_SIZE = 15
LABEL_FONT_SIZE = 30

@hydra.main(version_base=None, config_path='../../conf', config_name='dataset_variety')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')

    class_countss = []
    image_datasets = []
    for dataset_name in tqdm.tqdm(DATASET_NAMES, desc='loading images and class distributions'):
        class_counts = get_class_counts(dataset_name)
        class_countss.append(class_counts)

        image_dir = os.path.join(get_original_cwd_safe(), SAMPLES_PATH, dataset_name)
        image_dataset = [
            Image.open(os.path.join(image_dir, path))
            for path in os.listdir(image_dir)
        ]
        image_datasets.append(image_dataset)

    # creating the grid
    fig = plt.figure(figsize=(40, 12))
    gs = gridspec.GridSpec(2, len(DATASET_NAMES))

    for dataset_idx in range(len(DATASET_NAMES)):
        # images
        gs_imgs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, dataset_idx])

        # set title hack
        ax = plt.Subplot(fig, gs[0, dataset_idx])
        ax.set_title(DATASET_NAMES[dataset_idx], fontsize=LABEL_FONT_SIZE, pad=10)
        if dataset_idx == 0:
            ax.set_ylabel('image samples', fontsize=LABEL_FONT_SIZE, labelpad=50)

        # like ax.axis('off'), but doesn't hide ylabel
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        fig.add_subplot(ax)

        for image_idx, image in enumerate(image_datasets[dataset_idx]):
            ax = fig.add_subplot(gs_imgs[image_idx // 3, image_idx % 3])
            ax.imshow(image)
            ax.axis('off')

        # class distribution
        class_counts = class_countss[dataset_idx]

        class_labels = []
        class_counts = []
        for class_label in sorted(class_counts.keys()):
            class_labels.append(class_label)
            class_counts.append(class_counts[class_label])

        gs_dist = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, dataset_idx])
        ax = fig.add_subplot(gs_dist[0, 0])
        ax.bar(class_labels, class_counts, color='black')
        ax.tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE, rotation=35)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
        if dataset_idx == 0:
            ax.set_ylabel('image count', fontsize=LABEL_FONT_SIZE, labelpad=10)

    fig.tight_layout()
    plt.savefig('dataset_variety.pdf', format='pdf')
    plt.close()

 
if __name__ == '__main__':
    main()
