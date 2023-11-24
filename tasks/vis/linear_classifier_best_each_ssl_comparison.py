
import re
import os
import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs, dict_to_filters, get_best_criterium_filters
from src.models.build_data_path import get_vis_folder_path
from src.vis.constants import DATASETS_ORDER, REQUIRED_METRICS, REQUIRED_METRICS_SHORT_NAME, SSL_ORDER
from src.vis.helpers import mean_plus_minus_std, format_criterium, format_ssl, format_dataset
from src.utils.hydra import get_original_cwd_safe

plt.rcParams.update({'font.size': 22})

MODEL_TYPE_VERSION_PAIRS = [
    ('simclr', 'v1'),
    ('swav', 'v1'),
    ('dino', 'v2'),
]

@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_best_each_ssl_comparison')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')
    print(f'fetching results for {cfg.train_samples} train samples')

    if cfg.checkpoint_dir is None:
        # filters
        def get_ssl_filters(model_type, model_version):
            ssl_filters = []
            ssl_filters.append(get_best_criterium_filters(
                criterium='furthest',
                train_samples=cfg.train_samples,
                model_type=model_type,
                model_version=model_version,
            ))
            ssl_filters.append(get_best_criterium_filters(
                criterium='closest',
                train_samples=cfg.train_samples,
                model_type=model_type,
                model_version=model_version,
            ))
            ssl_filters.append(get_best_criterium_filters(
                criterium='half_in_half',
                train_samples=cfg.train_samples,
                model_type=model_type,
                model_version=model_version,
            ))
            ssl_filters.append(get_best_criterium_filters(
                criterium='fps',
                train_samples=cfg.train_samples,
                model_type=model_type,
                model_version=model_version,
            ))
            
            return ssl_filters

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

        filters = []
        for model_type, model_version in MODEL_TYPE_VERSION_PAIRS:
            filters.extend(get_ssl_filters(model_type, model_version))

        df_general = pd.concat(
            [
                get_runs('linear-classifier-soup', filters=f)
                for f in filters
            ], 
            ignore_index=True
        )
        df_general = transform_runs_general(df_general)
        print(f'total runs: {len(df_general)}')
        print()

        # saving the dataframes
        df_general.to_pickle('df_general.pickle')
    else:
        print(f'loading checkpoint from: {cfg.checkpoint_dir}')
        df_general = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general.pickle'))

    df_general = df_general[df_general['dataset'].isin(DATASETS_ORDER)]
    df_mean = df_general.groupby(['dataset', 'ssl', 'criterium'], as_index=False).mean()
    df_mean['ssl'] = df_mean['ssl'].apply(format_ssl)

    # creating the plots
    for metric_name in REQUIRED_METRICS_SHORT_NAME + ['classes']:
        max_rows = df_mean.groupby(['dataset', 'ssl'])[metric_name].idxmax()
        df_mean_max = df_mean.loc[max_rows]
        df_mean_max['dataset'] = df_mean_max['dataset'].apply(format_dataset)
        df_mean_max['SSL encoder'] = df_mean_max['ssl']

        fig = sns.catplot(
            data=df_mean_max,
            x='dataset', order=[format_dataset(d) for d in DATASETS_ORDER],
            hue='SSL encoder', hue_order=SSL_ORDER,
            palette='gray',
            y=metric_name,
            kind='bar',
            legend=False,
            aspect=2.0
        )
        plt.legend(loc='upper right')
        plt.xlabel(None)

        fig.savefig(f"{cfg.train_samples}_{metric_name}.pdf")
        plt.clf()

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
