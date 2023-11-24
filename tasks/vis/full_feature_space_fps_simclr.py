

import os
import hydra
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.hydra import get_original_cwd_safe
from src.utils.wandb import get_runs, dict_to_filters
from src.models.build_data_path import get_vis_folder_path


@hydra.main(version_base=None, config_path='../../conf', config_name='full_feature_space_fps_simclr')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')

    # defined fields
    required_metrics = [
        'test_accuracy_epoch_end',
        'test_f1_macro_epoch_end',
        'test_balanced_accuracy_epoch_end',
        'test_matthews_corrcoef_epoch_end',
        'test_cohen_kappa_score_epoch_end'
    ]
    required_metrics_name = [
        'ACC',
        'F1 MACRO',
        'BA',
        'MCC',
        'CKS'
    ]

    # filters
    filters_general = dict_to_filters({
        'training': {
            'train_samples': 100,
            'weights': {
                'type': 'simclr',
                'version': 'v1'
            }
        },
    })
    filters_general['config.use_scan_weights'] = {'$in': [None, False]}

    # fps l1
    filters_fps_l1 = dict_to_filters({})
    filters_fps_l1 = {**filters_general, **filters_fps_l1}

    # l2
    filters_fps_l2 = dict_to_filters({
        'kmeans': {
            'clusters': 1,
            'criterium': 'fps'
        },
        'features': {
            'scaling': 'standard'
        }
    })
    filters_fps_l2 = {**filters_general, **filters_fps_l2}

    # fps cosine
    filters_fps_cosine = dict_to_filters({})
    filters_fps_cosine = {**filters_general, **filters_fps_cosine}

    # random
    filters_random = dict_to_filters({
        'kmeans': {
            'clusters': 1,
            'criterium': 'random'
        }
    })
    filters_random = {**filters_general, **filters_random}

    # ceiling
    filters_ceiling = dict_to_filters({
        'training': {
            'train_samples': None,
            'weights': {
                'type': 'simclr',
                'version': 'v1'
            }
        },
    })
    filters_ceiling['config.use_scan_weights'] = {'$in': [None, False]}

    # transforms
    def transform_runs(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)
        
        df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    # fetching all runs
    df_fps_l1 = get_runs('simclr-fps-l1', filters=filters_fps_l1) 
    df_fps_l1 = transform_runs(df_fps_l1)
    print('df_fps_l1')
    print(df_fps_l1)
    print()

    df_fps_l2 = get_runs('badge-sampling', filters=filters_fps_l2) 
    df_fps_l2 = transform_runs(df_fps_l2)
    print('df_fps_l2 ')
    print(df_fps_l2 )
    print()

    df_fps_cosine = get_runs('simclr-fps-cosine', filters=filters_fps_cosine)
    df_fps_cosine = transform_runs(df_fps_cosine)
    print('df_fps_cosine')
    print(df_fps_cosine)
    print()

    df_random = get_runs('badge-sampling', filters=filters_random)
    df_random = transform_runs(df_random)
    print('df_random')
    print(df_random)
    print()

    df_ceiling = get_runs('badge-sampling', filters=filters_ceiling)
    df_ceiling = transform_runs(df_ceiling)
    print('df_ceiling')
    print(df_ceiling)
    print()

    # concatenating
    df_fps_l1['method'] = 'fps-l1'
    df_fps_l2['method'] = 'fps-l2'
    df_fps_cosine['method'] = 'fps-cosine'
    df_random['method'] = 'random'
    df_fps = pd.concat([df_fps_l1, df_fps_l2, df_fps_cosine, df_random])

    # creating the plots
    dataset_order = ['matek', 'isic', 'retinopathy', 'jurkat', 'cifar10']
    for metric_name in required_metrics_name:
        plt.style.use('ggplot')

        g = sns.catplot(
            df_fps,
            hue='method',
            y=metric_name,
            col='dataset',
            kind='strip',
            col_order=dataset_order,
            dodge=True,
            # hue_order=['fps-l1', 'fps-l2', 'fps-cosine'],
            # dodge=True, alpha=.25, zorder=1, legend=False
        )

        fig, axes = g.fig, g.axes

        # Add horizontal line to each subplot
        for i, ax in enumerate(axes[0]):
            y = df_ceiling[df_ceiling['dataset'] == dataset_order[i]][metric_name].item()
            ax.axhline(y=y, color='r', linestyle='--')

        # for i, ax in enumerate(axes[0]):
        #     y = df_ceiling[df_ceiling['dataset'] == dataset_order[i]][metric_name].item()
        #     ax.axhline(y=y, color='r', linestyle='--')

        # sns.stripplot()

        # sns.pointplot(
        #     data=df_fps,
        #     x='dataset', y=metric_name, hue='method',
        #     join=False, dodge=.8 - .8 / 3, palette='dark',
        #     order=['matek', 'isic', 'retinopathy', 'jurkat', 'cifar10'],
        #     hue_order=['fps-l1', 'fps-l2', 'fps-cosine'],
        #     markers='d', scale=.75, errorbar=None
        # )

        # sns.lineplot(
        #     data=df_ceiling,
        #     x='dataset', y=metric_name,
        #     # order=['matek', 'isic', 'retinopathy', 'jurkat', 'cifar10'],
        # )

        # Improve the legend
        # sns.move_legend(
        #     ax, loc='lower right', ncol=3, frameon=True, columnspacing=1, handletextpad=0
        # )

        g.savefig(f'{metric_name}.pdf')

    print(f'everything saved to: {os.getcwd()}')


if __name__ == '__main__':
    main()
