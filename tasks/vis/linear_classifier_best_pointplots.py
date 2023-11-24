

import os
import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs, dict_to_filters
from src.models.build_data_path import get_vis_folder_path
from src.vis.constants import DATASETS_ORDER, CRITERIA_BEST_ORDER, CRITERIA_BEST_ORDER_NO_RANDOM, \
                              REQUIRED_METRICS, REQUIRED_METRICS_SHORT_NAME, REQUIRED_METRICS_FULL_NAME
from src.vis.helpers import mean_plus_minus_std, format_criterium
from src.utils.hydra import get_original_cwd_safe


@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_best_pointplots')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')
    print(f'fetching results for {cfg.train_samples} train samples')

    if cfg.checkpoint_dir is None:
        # filters
        filters_fps = dict_to_filters({
            'training': {
                'train_samples': cfg.train_samples,
                'oversample': True,
                'weights': {
                    'freeze': True,
                },
            },
            'features': {
                'scaling': 'standard'
            },
            'kmeans': {
                'mode': 'kmeans',
                'clusters': 1,
                'criterium': 'fps'
            },
            'use_scan_weights': False,
        })
        filters_random = dict_to_filters({
            'training': {
                'train_samples': cfg.train_samples,
                'oversample': True,
                'weights': {
                    'freeze': True,
                },
            },
            'features': {
                'scaling': 'standard'
            },
            'kmeans': {
                'mode': 'kmeans',
                'clusters': 1,
                'criterium': 'random'
            },
            'use_scan_weights': False,
        })
        filters_closest = dict_to_filters({
            'training': {
                'train_samples': cfg.train_samples,
                'oversample': True,
                'weights': {
                    'freeze': True,
                },
            },
            'features': {
                'scaling': 'standard'
            },
            'kmeans': {
                'mode': 'kmeans',
                'clusters': cfg.train_samples,
                'criterium': 'closest'
            },
            'use_scan_weights': False,
        })
        filters_cold_paws = dict_to_filters({
            'training': {
                'train_samples': cfg.train_samples,
                'oversample': True,
                'weights': {
                    'freeze': True,
                },
            },
            'use_scan_weights': False,
        })
        filters_ceiling = dict_to_filters({
            'training': {
                # 'train_samples': None,
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
            df['ssl'] = df.apply(
                lambda r: r['config']['training']['weights']['type'] + '/' + r['config']['training']['weights']['version'],
                axis=1
            ) 

            df = df.drop(['name', 'summary', 'config'], axis=1)
            df = df.dropna()

            return df

        print(f'fetching fps, closest ({cfg.train_samples}), and random runs')
        df_general = pd.concat([
            get_runs('linear-classifier-soup', filters=filters_fps),
            get_runs('linear-classifier-soup', filters=filters_closest),
            get_runs('linear-classifier-soup', filters=filters_random),
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
        df_ceiling = get_runs('linear-classifier-ceiling', filters=filters_ceiling)
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
    df_general = pd.concat([df_general, df_cold_paws], ignore_index=True)

    # processing dataframes
    df_general_melt = df_general.melt(
        id_vars=['dataset', 'ssl', 'criterium'],
        var_name='metric_name',
        value_name='metric_value',
    )

    # creating the plots
    row_order = REQUIRED_METRICS_SHORT_NAME
    col_order = DATASETS_ORDER
    x_order = CRITERIA_BEST_ORDER_NO_RANDOM
    hue_order = CRITERIA_BEST_ORDER_NO_RANDOM
    # for ssl in df_general_melt['ssl'].unique():
    # for ssl in ['simclr/v1', 'swav/v1']:
    for ssl in ['simclr/v1']:
        df_ssl = df_general_melt[df_general_melt['ssl'] == ssl]

        fig = sns.catplot(
            data=df_ssl,
            x='criterium', order=x_order,
            sharex=False,
            hue='criterium', hue_order=hue_order,
            palette='gray',
            y='metric_value',
            row='metric_name', row_order=row_order,
            col='dataset', col_order=col_order,
            kind='point',
            errorbar='sd',
            aspect=.3,
            capsize=.5,
            scale=1.5,
            markers='_',
            facet_kws={
                'ylim': (0.0, 1.0),
            },
        )

        # baseline
        ax = None
        for i, row in enumerate(fig.axes):
            metric_name = row_order[i]

            for j, ax in enumerate(row):
                dataset = col_order[j]

                # baseline
                y = df_ceiling[
                    (df_ceiling['dataset'] == dataset) &
                    (df_ceiling['ssl'] == ssl)
                ][metric_name]

                if len(y) != 1:
                    continue

                y = y.item()

                ax.axhline(y, color='black',  linestyle='solid', label='baseline')

                # random
                df_random = df_ssl[
                    (df_ssl['dataset'] == dataset) &
                    (df_ssl['criterium'] == 'random') &
                    (df_ssl['metric_name'] == metric_name)
                ]['metric_value']
                random_mean = df_random.mean()
                random_std = df_random.std()

                ax.axhline(y=random_mean + random_std, color='black', linestyle='dashed')
                ax.axhline(y=random_mean - random_std, color='black', linestyle='dashed', label='random')

                if j == 0:
                    ax.set_ylabel(metric_name)
                else:
                    ax.set_ylabel(None)

                # if i == 0:
                #     ax.set_title(dataset)
                # else:
                #     ax.set_title(None)
                ax.set_title(dataset)

                # creating the legend
                # if j == (len(col_order) // 2):
                if j == len(col_order) - 1:
                    handles, labels = ax.get_legend_handles_labels()
                    labels = [format_criterium(l, train_samples=cfg.train_samples) for l in labels]

                    # leg_criteria = ax.legend(
                    #     handles=handles[:3], labels=labels[:3], 
                    #     loc='lower center', bbox_to_anchor=(1, 0),
                    #     framealpha=1.0, edgecolor=(0,0,0,0),
                    #     ncol=3
                    # )
                    # leg_baselines = ax.legend(
                    #     handles=handles[3:], labels=labels[3:], 
                    #     framealpha=1.0, edgecolor=(0,0,0,0),
                    #     loc='lower center', bbox_to_anchor=(1, 0.1)
                    # )
                    # ax.add_artist(leg_criteria)
                    ax.legend(
                        handles=handles, labels=labels, 
                        ncols=len(handles),
                        loc='lower right', bbox_to_anchor=(.8,0),
                        framealpha=1.0, 
                        fancybox=False, edgecolor='black',
                    )

                ax.set_xticks([])

                # x_tick_labels = [
                #     format_criterium(l, train_samples=cfg.train_samples)
                #     for l in x_order
                # ]
                # ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
                ax.set_xlabel(None)

        # creating the legend
        # handles, labels = ax.get_legend_handles_labels()
        # labels = [l.replace('_', ' ') for l in labels]
        # fig.add_legend(handles=reversed(handles), labels=reversed(labels))

        # saving images
        fig.savefig(f"{cfg.train_samples}_{ssl.replace('/', '_')}.pdf")
        plt.clf()

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
