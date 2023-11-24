
import glob
import os
import hydra
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
import numpy as np
from PIL import Image
from matplotlib import gridspec
from omegaconf import DictConfig, OmegaConf
from src.datasets.datasets import get_dataset_class_by_name
from src.utils.hydra import get_original_cwd_safe
from src.vis.helpers import get_class_counts, get_class_examples, get_grayscale_color
from collections import defaultdict

plt.rcParams.update({'font.size': 12})

DATASETS = ['matek', 'isic', 'retinopathy', 'jurkat', 'cifar10']

@hydra.main(version_base=None, config_path='../../conf', config_name='dataset_introduction')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')

    if cfg.seed is not None:
        print(f'setting random seed to {cfg.seed}')
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # creating needed data for each dataset
    dataset_datas = {}
    for dataset in DATASETS:
        class_counts = get_class_counts(dataset)
        class_examples = get_class_examples(dataset)
        class_order = sorted(
            class_counts.keys(),
            key=lambda k: class_counts[k], 
            reverse=True
        )
        total_images = sum(class_counts.values())
        class_count_colors = {
            # class_: 1.0 - class_counts[class_] / total_images
            class_: 1.0 - np.power(class_counts[class_], .5) / np.sum(np.power(list(class_counts.values()), .5))
            for class_ in class_counts.keys()
        }
        class_text_colors = {
            class_: 1.0 if class_count_colors[class_] < 0.5 else 0.1
            for class_ in class_counts.keys()
        }

        data = {
            'dataset': dataset,
            'class_counts': class_counts,
            'class_examples': class_examples,
            'class_order': class_order,
            'total_images': total_images,
            'class_count_colors': class_count_colors,
            'class_text_colors': class_text_colors,
        }
        dataset_datas[dataset] = data

    max_classes = max(len(d['class_counts']) for d in dataset_datas.values())

    # creating the grid
    rows = 2 * len(DATASETS)
    cols = max_classes
    coef = 1.0
    fig = plt.figure(figsize=(coef * cols, coef * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.1, hspace=0.1)

    # You can now customize each subplot or plot data in each subplot
    for i in range(rows):
        # unraveling dataset data
        dataset_data = dataset_datas[DATASETS[i // 2]]

        dataset = dataset_data['dataset']
        class_counts = dataset_data['class_counts']
        class_examples = dataset_data['class_examples']
        class_order = dataset_data['class_order']
        total_images = dataset_data['total_images']
        class_count_colors = dataset_data['class_count_colors']
        class_text_colors = dataset_data['class_text_colors']

        for j in range(cols):
            ax = fig.add_subplot(gs[i, j])

            if j >= len(class_counts):
                ax.axis('off')
                continue

            class_ = class_order[j]

            # setting titles
            if (i % 2) == 0:
                ax.set_title(class_)

            # removing axes while keeping ylabel
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if j == 0:
                if (i % 2) == 0:
                    ax.set_ylabel('example')
                elif (i % 2) == 1:
                    # ax.set_ylabel('count', loc='top')
                    ax.set_ylabel('num', loc='top')

            if (i % 2) == 0:
                # image example
                if dataset == 'jurkat':
                    image = Image.open(class_examples[class_])
                    image_array = np.array(image)
                    red_channel = image_array[:, :, 2]
                    ax.imshow(red_channel, cmap="gray")
                else:
                    image = Image.open(class_examples[class_])
                    ax.imshow(image)
            elif (i % 2) == 1:
                # class count
                rect = patches.Rectangle(
                    (0.0, 0.5), 
                    1.0, 0.5, 
                    linewidth=1, 
                    edgecolor=get_grayscale_color(max(class_count_colors[class_] - 0.1, 0.0)),
                    facecolor=get_grayscale_color(class_count_colors[class_])
                )
                ax.add_patch(rect)

                # Add text in the middle of the rectangle
                ax.text(
                    0.5, 0.75, 
                    class_counts[class_], 
                    ha='center', va='center', 
                    color=get_grayscale_color(class_text_colors[class_]), 
                    # fontsize=11
                )

    # Adjust layout to prevent overlap
    # fig.tight_layout()
    # plt.tight_layout()
    fig.savefig('datasets.pdf', format='pdf')

 
if __name__ == '__main__':
    main()
