
import re
import os
import glob
import hydra
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from src.models.build_data_path import get_data_folder_path

DATASET_NAMES = [
    'matek',
    'isic',
    'retinopathy',
    'jurkat',
    'cifar10'
]

@hydra.main(version_base=None, config_path='../../conf', config_name='datasets_clusters_sweeps')
def main(cfg: DictConfig):
    print(f'saving everything to: {os.getcwd()}')

    metric = cfg.metric

    unique_models = get_unique_models()
    print(f'unique models found: {unique_models}')

    figsize = (len(DATASET_NAMES),len(unique_models))
    _, plots = plt.subplots(len(unique_models), len(DATASET_NAMES), sharex=True, sharey=True, figsize=figsize)
    for i in range(len(unique_models)):
        model_type, model_version = unique_models[i]
        for j in range(len(DATASET_NAMES)):
            dataset_name = DATASET_NAMES[j]
            
            figure_path = get_clusters_sweep_figure_path(
                dataset_name, model_type, model_version, metric)

            plot = plots[i][j]

            if figure_path is not None:
                image = convert_from_path(figure_path, dpi=300)[0]
                plot.imshow(image, interpolation=None)

            # make xaxis invisibel
            # plot.xaxis.set_visible(False)
            # make spines (the box) invisible
            plt.setp(plot.spines.values(), visible=False)
            # remove ticks and labels for the left axis
            plot.tick_params(left=False, labelleft=False)
            plot.tick_params(bottom=False, labelbottom=False)
            #remove background patch (only needed for non-white background)
            plot.patch.set_visible(False)

    # setting overall axes ticks
    for i in range(len(unique_models)):
        model_type, model_version = unique_models[i]
        plt.setp(plots[i, 0], ylabel=f'{model_type}/{model_version}')

    for i in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[i]
        plt.setp(plots[-1, i], xlabel=dataset_name)

    plt.savefig(f'datasets_clusters_sweeps_{metric}.pdf', bbox_inches='tight', dpi=1000)


def get_clusters_sweep_figure_path(dataset_name, model_type, model_version, metric):
    vis_folder_path = os.path.join(
        get_data_folder_path(), dataset_name, model_type, model_version, 'clusters_sweep')

    figure_paths = glob.glob(f'{vis_folder_path}/*{metric}.pdf')

    return figure_paths[0] if len(figure_paths) > 0 else None


def get_unique_models():
    # e.g. src/models/data/retinopathy/swav/v1
    model_paths = glob.glob(f'{get_data_folder_path()}/*/*/*')

    # e.g. (swav, v1)
    models = [re.search(r'([^/]+)/([^/]+)$', path).groups() for path in model_paths]
    unique_models = sorted(list(set(models)))

    return unique_models


if __name__ == '__main__':
    main()
