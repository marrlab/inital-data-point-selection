

import os
import hydra
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs, dict_to_filters, get_best_criterium_filters
from src.models.build_data_path import get_vis_folder_path
from src.vis.constants import DATASETS_ORDER, CRITERIA_ORDER_NO_RANDOM, REQUIRED_METRICS, REQUIRED_METRICS_SHORT_NAME 
from src.vis.helpers import mean_plus_minus_std, format_criterium
from src.utils.hydra import get_original_cwd_safe


@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_best_spider')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')
    print(f'fetching results for {cfg.train_samples} train samples')

    if cfg.checkpoint_dir is None:
        # filters
        filters_random = get_best_criterium_filters(
            criterium='random',
            train_samples=cfg.train_samples,
        )
        filters_cold_paws = get_best_criterium_filters(
            criterium='cold_paws',
            train_samples=cfg.train_samples,
        )
        filters_furthest = get_best_criterium_filters(
            criterium='furthest',
            train_samples=cfg.train_samples,
        )
        filters_closest = get_best_criterium_filters(
            criterium='closest',
            train_samples=cfg.train_samples,
        )
        filters_half_in_half = get_best_criterium_filters(
            criterium='half_in_half',
            train_samples=cfg.train_samples,
        )
        filters_fps = get_best_criterium_filters(
            criterium='fps',
            train_samples=cfg.train_samples,
        )
        filters_ceiling = dict_to_filters({
            'training': {
                'weights': {
                    'freeze': True,
                }
            },
            'use_scan_weights': False,
        })

        def transform_runs_general(df):
            if len(df) == 0:
                return pd.DataFrame()

            for i in range(len(REQUIRED_METRICS)):
                df[REQUIRED_METRICS_SHORT_NAME[i]] = \
                    df.apply(lambda r: r['summary'].get(REQUIRED_METRICS[i], None), axis=1)

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['classes'] = df.apply(
                lambda r: r['config']['classes'], axis=1)
            df['criterium'] = df.apply(
                lambda r: r['config']['kmeans']['criterium'], axis=1)
            df['ssl'] = df.apply(
                lambda r: r['config']['training']['weights']['type'] + '/' + r['config']['training']['weights']['version'],
                axis=1
            ) 

            df = df.drop(['name', 'summary', 'config'], axis=1)
            df = df.dropna()

            return df

        def transform_runs_cold_paws(df):
            if len(df) == 0:
                return pd.DataFrame()

            for i in range(len(REQUIRED_METRICS)):
                df[REQUIRED_METRICS_SHORT_NAME[i]] = \
                    df.apply(lambda r: r['summary'].get(REQUIRED_METRICS[i], None), axis=1)

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['classes'] = df.apply(
                lambda r: r['config']['classes'], axis=1)
            df['criterium'] = 'cold_paws'
            df['ssl'] = df.apply(
                lambda r: r['config']['training']['weights']['type'] + '/' + r['config']['training']['weights']['version'],
                axis=1
            ) 

            df = df.drop(['name', 'summary', 'config'], axis=1)
            df = df.dropna()

            return df

        def transform_runs_ceiling(df):
            if len(df) == 0:
                return pd.DataFrame()

            for i in range(len(REQUIRED_METRICS)):
                df[REQUIRED_METRICS_SHORT_NAME[i]] = \
                    df.apply(lambda r: r['summary'].get(REQUIRED_METRICS[i], None), axis=1)

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['classes'] = df.apply(
                lambda r: r['config']['dataset']['num_classes'], axis=1)
            df['criterium'] = 'full_data'
            df['ssl'] = df.apply(
                lambda r: r['config']['training']['weights']['type'] + '/' + r['config']['training']['weights']['version'],
                axis=1
            ) 

            df = df.drop(['name', 'summary', 'config'], axis=1)
            df = df.dropna()

            return df

        print(f'fetching fps, closest ({cfg.train_samples}), and random runs')
        df_general = pd.concat([
            get_runs('linear-classifier-soup', filters=filters_random),
            get_runs('linear-classifier-soup', filters=filters_furthest),
            get_runs('linear-classifier-soup', filters=filters_half_in_half),
            get_runs('linear-classifier-soup', filters=filters_closest),
            get_runs('linear-classifier-soup', filters=filters_fps),
        ], ignore_index=True)
        df_general = transform_runs_general(df_general)
        print(f'total runs: {len(df_general)}')
        print()

        print('fetching all cold paws runs')
        df_cold_paws = get_runs('linear-classifier-cold-paws-soup', filters=filters_cold_paws)
        df_cold_paws = transform_runs_cold_paws(df_cold_paws)
        print(f'total runs: {len(df_cold_paws)}')
        print()

        print('fetching all ceiling runs')
        df_ceiling = get_runs('linear-classifier-ceiling-best', filters=filters_ceiling)
        df_ceiling = transform_runs_ceiling(df_ceiling)
        print(f'total runs: {len(df_ceiling)}')
        print()

        # saving the dataframes
        df_general.to_pickle('df_general.pickle')
        df_cold_paws.to_pickle('df_cold_paws.pickle')
        df_ceiling.to_pickle('df_ceiling.pickle')
    else:
        print(f'loading checkpoint from: {cfg.checkpoint_dir}')
        df_general = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general.pickle'))
        df_cold_paws = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_cold_paws.pickle'))
        df_ceiling = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_ceiling.pickle'))

    # merging dataframes
    df_general = pd.concat([df_general, df_cold_paws, df_ceiling], ignore_index=True)
    df_mean = df_general.groupby(['dataset', 'criterium', 'ssl'], as_index=False).mean()

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [
        n / float(len(REQUIRED_METRICS_SHORT_NAME)) * 2 * np.pi 
        for n in range(len(REQUIRED_METRICS_SHORT_NAME))
    ]
    angles += angles[:1]

    for ssl in ['simclr/v1']:
        fig, axes = plt.subplots(1, len(DATASETS_ORDER), subplot_kw=dict(projection='polar'), figsize=(15,3))
        handles = []
        labels = []
        for i, dataset in enumerate(DATASETS_ORDER):
            df_mean_cur = df_mean[
                (df_mean['ssl'] == ssl) &
                (df_mean['dataset'] == dataset)
            ]

            ax = axes[i]
            ax.set_title(dataset, pad=15)
 
            # If you want the first axis to be on top:
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
 
            # Draw one axe per variable + add labels
            ax.set_xticks(angles[:-1], REQUIRED_METRICS_SHORT_NAME)

            # Draw ylabels
            df_mean_cur_metric = df_mean_cur[REQUIRED_METRICS_SHORT_NAME]
            y_min = df_mean_cur_metric.min().min()
            y_min = np.floor(10 * y_min) / 10
            y_max = df_mean_cur_metric.max().max()
            y_max = np.ceil(10 * y_max) / 10
            y_mid = y_min + (y_max - y_min) / 2
            y_mid = np.round(10 * y_mid) / 10

            yticks = [y_min, y_mid, y_max]
            ax.set_rlabel_position(0)
            ax.set_yticks(yticks, yticks, size=10)
            ax.set_ylim(y_min, y_max)
            ax.set_axisbelow(False)

            # random 
            values = df_mean_cur[df_mean_cur['criterium'] == 'random'][REQUIRED_METRICS_SHORT_NAME].values.flatten().tolist()
            values += values[:1]
            handle, = ax.plot(angles, values, color='black', linewidth=1, linestyle='dotted', label='random')
            if i == 0:
                handles.append(handle)
                labels.append('random')
 
            # draw polygons
            for criterium in CRITERIA_ORDER_NO_RANDOM:
                values = df_mean_cur[df_mean_cur['criterium'] == criterium][REQUIRED_METRICS_SHORT_NAME].values.flatten().tolist()
                values += values[:1]
                handle, = ax.plot(angles, values, linewidth=1, linestyle='solid', label=criterium)

                if i == 0:
                    handles.append(handle)
                    labels.append(format_criterium(criterium, train_samples=cfg.train_samples))

            # ceiling
            values = df_mean_cur[df_mean_cur['criterium'] == 'full_data'][REQUIRED_METRICS_SHORT_NAME].values.flatten().tolist()
            values += values[:1]
            handle, = ax.plot(angles, values, color='black', linewidth=1, linestyle='dashed', label='ceiling')
            if i == 0:
                handles.append(handle)
                labels.append('full data')

        # saving images
        plt.subplots_adjust(wspace=.7)
        fig.legend(handles=handles, labels=labels, loc='lower center', ncols=10, borderaxespad=0.1)

        plt.savefig(f"{cfg.train_samples}_{ssl.replace('/', '_')}.pdf")
        plt.clf()

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
