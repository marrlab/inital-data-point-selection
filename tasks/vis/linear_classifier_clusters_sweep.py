

import os
import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs, dict_to_filters
from src.utils.hydra import get_original_cwd_safe
from src.vis.helpers import format_criterium
from src.models.build_data_path import get_vis_folder_path
from src.vis.constants import DATASETS_ORDER, OUR_CRITERIA_ORDER_NO_RANDOM


@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_clusters_sweep')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')
    print(f'fetching results for {cfg.train_samples} train samples')

    # defined fields
    required_metrics = [
        'test_f1_macro_epoch_end',
        'test_accuracy_epoch_end',
        'test_balanced_accuracy_epoch_end',
        'test_matthews_corrcoef_epoch_end',
        'test_cohen_kappa_score_epoch_end',
    ]
    required_metrics_name = [
        'F1-MACRO',
        'ACC',
        'BACC',
        'MCC',
        'KAPPA',
    ]

    if cfg.checkpoint_dir is None:
        # filters
        filters_general = dict_to_filters({
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
            },
            'use_scan_weights': False,
        })
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

            for i in range(len(required_metrics)):
                df[required_metrics_name[i]] = \
                    df.apply(lambda r: r['summary'].get(required_metrics[i], None), axis=1)

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['classes'] = df.apply(
                lambda r: r['config']['classes'], axis=1)
            df['clusters'] = df.apply(
                lambda r: r['config']['kmeans']['clusters'], axis=1)
            df['criterium'] = df.apply(
                lambda r: r['config']['kmeans']['criterium'], axis=1)
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

            for i in range(len(required_metrics)):
                df[required_metrics_name[i]] = \
                    df.apply(lambda r: r['summary'].get(required_metrics[i], None), axis=1)

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

        print('fetching all general runs')
        df_general = get_runs('linear-classifier-soup', filters=filters_general)
        df_general = transform_runs_general(df_general)
        print(f'total runs: {len(df_general)}')
        print()

        print('fetching all ceiling runs')
        df_ceiling = get_runs('linear-classifier-ceiling-best', filters=filters_ceiling)
        df_ceiling = transform_runs_ceiling(df_ceiling)
        print(f'total runs: {len(df_ceiling)}')
        print()

        # saving the dataframes
        df_general.to_pickle('df_general.pickle')
        df_ceiling.to_pickle('df_ceiling.pickle')
    else:
        print(f'loading checkpoint from: {cfg.checkpoint_dir}')
        df_general = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general.pickle'))
        df_ceiling = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_ceiling.pickle'))

    # processing dataframes
    df_general = df_general.melt(
        id_vars=['dataset', 'ssl', 'criterium', 'clusters'],
        var_name='metric_name',
        value_name='metric_value',
    )

    # creating the plots
    # row_order = ['F1-MACRO', 'ACC']
    row_order = ['F1-MACRO']
    col_order = DATASETS_ORDER 
    hue_order = OUR_CRITERIA_ORDER_NO_RANDOM
    # for ssl in df_general['ssl'].unique():
    for ssl in ['simclr/v1']:
        df_ssl = df_general[df_general['ssl'] == ssl]
        df_ssl = df_ssl.drop(['ssl'], axis=1)

        # hacky way to obtain section of the legend
        handles_criteria, labels_criteria = sns.relplot(
            data=df_ssl,
            x='clusters',
            hue='criterium', hue_order=hue_order,
            palette='colorblind',
            y='metric_value',
            row='metric_name', row_order=row_order,
            col='dataset', col_order=col_order,
            aspect=.5,
            kind='line',
            errorbar='sd',
            facet_kws={
                'ylim': (0.0, 0.7),
            },
        ).axes[0][0].get_legend_handles_labels()
        plt.clf()

        fig = sns.relplot(
            data=df_ssl,
            x='clusters',
            hue='criterium', hue_order=hue_order,
            palette='colorblind',
            y='metric_value',
            row='metric_name', row_order=row_order,
            col='dataset', col_order=col_order,
            aspect=.5,
            kind='line',
            errorbar='sd',
            facet_kws={
                'ylim': (0.0, 0.7),
            },
            legend=False
        )

        # baseline
        for i, row in enumerate(fig.axes):
            metric_name = row_order[i]

            for j, ax in enumerate(row):
                dataset = col_order[j]

                # baseline
                df_ceiling_cur = df_ceiling[
                    (df_ceiling['dataset'] == dataset) &
                    (df_ceiling['ssl'] == ssl)
                ][metric_name]

                ceiling_mean = df_ceiling_cur.mean()
                ceiling_std = df_ceiling_cur.std()

                ax.axhline(y=ceiling_mean - ceiling_std, color='black', linestyle='dashed', label='full data')
                ax.axhline(y=ceiling_mean + ceiling_std, color='black', linestyle='dashed')

                # random
                df_random = df_ssl[
                    (df_ssl['dataset'] == dataset) &
                    (df_ssl['criterium'] == 'random') &
                    (df_ssl['metric_name'] == metric_name)
                ]['metric_value']
                random_mean = df_random.mean()
                random_std = df_random.std()

                # ax.axhline(y=random_mean, color='black', linestyle='dashed', label='random')
                ax.axhline(y=random_mean - random_std, color='black', linestyle='dotted', label='random')
                ax.axhline(y=random_mean + random_std, color='black', linestyle='dotted')

                if j == 0:
                    ax.set_ylabel(metric_name)
                else:
                    ax.set_ylabel(None)

                if i == 0:
                    ax.set_title(dataset)
                else:
                    ax.set_title(None)

                if cfg.show_legend and j == len(col_order) - 1:
                    # creating the legend
                    handles_baseline, labels_baseline = fig.axes[0][0].get_legend_handles_labels()
                    labels_criteria = [format_criterium(l, train_samples=None) for l in labels_criteria]

                    handles = handles_baseline + handles_criteria
                    labels = labels_baseline + labels_criteria

                    # putting full data at the end of the legend
                    handles = handles[1:] + handles[:1]
                    labels = labels[1:] + labels[:1]

                    ax.legend(
                        handles=handles, labels=labels,
                        ncols=len(handles),
                        loc='lower right', bbox_to_anchor=(0.95,0),
                        framealpha=1.0, 
                        fancybox=False, edgecolor='black',
                    )
        
        fig.savefig(f"{cfg.train_samples}_{ssl.replace('/', '_')}.pdf")
        plt.clf()

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
