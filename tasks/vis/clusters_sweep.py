

import os
import hydra
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs, dict_to_filters
from src.models.build_data_path import get_vis_folder_path


@hydra.main(version_base=None, config_path='../../conf', config_name='clusters_sweep')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    folder_path = get_vis_folder_path(cfg)
    print(f'saving everything to: {folder_path}')

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

    # general
    filters_general = dict_to_filters({
        'dataset.name': cfg_dict['dataset']['name'],
        'training': cfg_dict['training'],
        'features': cfg_dict['features'],
        'use_scan_weights': cfg_dict['use_scan_weights']
    })

    def transform_runs_general(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)

        df['classes'] = df.apply(
            lambda r: r['config']['classes'], axis=1)
        df['clusters'] = df.apply(
            lambda r: r['config']['kmeans']['clusters'], axis=1)
        df['criterium'] = df.apply(
            lambda r: r['config']['kmeans']['criterium'], axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    # random baseline
    filters_random = dict_to_filters({
        'kmeans': {
            'clusters': 1,
            'criterium': 'random'
        }
    })
    filters_random = {**filters_general, **filters_random}

    df_random = get_runs('badge-sampling', filters=filters_random)
    df_random = transform_runs_general(df_random)
    print(f'random runs: {len(df_random)}')

    # badge sampling
    filters_badge = dict_to_filters({
        'kmeans': {
            'mode': 'kmeans'
        }
    })
    filters_badge = {**filters_general, **filters_badge}

    df_badge = get_runs('badge-sampling', filters=filters_badge)
    df_badge = transform_runs_general(df_badge)
    print(f"other runs: \n{df_badge.groupby(['criterium', 'clusters']).size()}")

    # saving the dataframes
    df_random.to_pickle('df_random.pickle')
    df_badge.to_pickle('df_badge.pickle')

    # creating the plots
    for metric_name in required_metrics_name + ['classes']:
        random_mean = df_random[metric_name].mean()
        random_std = df_random[metric_name].std()

        fig, ax = plt.subplots()
        ax.set_xticks(df_badge['clusters'].unique())
        ax.axvline(cfg.clusters_highlight, color='black',
                   linestyle='--', label='number of classes')
        sns.lineplot(data=df_badge, x='clusters', y=metric_name, hue='criterium', hue_order=[
                     'random', 'closest', 'furthest', 'half_in_half', 'fps'], errorbar='sd', marker='o', ax=ax)
        ax.axhline(random_mean, color='black', label='random baseline')
        ax.axhspan(random_mean - random_std, random_mean +
                   random_std, color='black', alpha=0.1)
        ax.legend()

        fig.savefig(os.path.join(folder_path, f'{cfg.dataset.name}_{metric_name}.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    main()
