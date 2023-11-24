
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
from src.vis.constants import DATASETS_ORDER, CRITERIA_ORDER, REQUIRED_METRICS, REQUIRED_METRICS_SHORT_NAME 
from src.vis.helpers import mean_plus_minus_std, format_criterium
from src.utils.hydra import get_original_cwd_safe


@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_best_tables')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')
    print(f'fetching results for {cfg.train_samples} train samples')

    if cfg.checkpoint_dir is None:
        # filters
        filters_random = get_best_criterium_filters(
            criterium='random',
            train_samples=cfg.train_samples,
            model_type=cfg.model_type,
            model_version=cfg.model_version,
        )
        filters_cold_paws = get_best_criterium_filters(
            criterium='cold_paws',
            train_samples=cfg.train_samples,
            model_type=cfg.model_type,
            model_version=cfg.model_version,
        )
        filters_furthest = get_best_criterium_filters(
            criterium='furthest',
            train_samples=cfg.train_samples,
            model_type=cfg.model_type,
            model_version=cfg.model_version,
        )
        filters_closest = get_best_criterium_filters(
            criterium='closest',
            train_samples=cfg.train_samples,
            model_type=cfg.model_type,
            model_version=cfg.model_version,
        )
        filters_half_in_half = get_best_criterium_filters(
            criterium='half_in_half',
            train_samples=cfg.train_samples,
            model_type=cfg.model_type,
            model_version=cfg.model_version,
        )
        filters_fps = get_best_criterium_filters(
            criterium='fps',
            train_samples=cfg.train_samples,
            model_type=cfg.model_type,
            model_version=cfg.model_version,
        )
        filters_ceiling = dict_to_filters({
            'training': {
                'weights': {
                    'type': cfg.model_type,
                    'version': cfg.model_version,
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

    # creating the tables
    for ssl in df_general['ssl'].unique():
        # creating the table
        for metric_name in REQUIRED_METRICS_SHORT_NAME + ['classes']:
            df_ssl = df_general[df_general['ssl'] == ssl][['dataset', 'criterium', metric_name]]

            # mean
            df_mean = df_ssl.groupby(['dataset', 'criterium'], as_index=False).mean()
            df_mean = df_mean.pivot(index='dataset', columns='criterium')[metric_name]
            df_mean = df_mean[CRITERIA_ORDER + ['full_data']]
            df_mean = df_mean.loc[DATASETS_ORDER]
            df_mean = df_mean.T

            # std
            df_std = df_ssl.groupby(['dataset', 'criterium'], as_index=False).std()
            df_std = df_std.pivot(index='dataset', columns='criterium')[metric_name]
            df_std = df_std[CRITERIA_ORDER + ['full_data']]
            df_std = df_std.loc[DATASETS_ORDER]
            df_std = df_std.T

            decimal = 1 if metric_name == 'classes' else 2
            df_table = mean_plus_minus_std(df_mean=df_mean, df_std=df_std, decimal=decimal)

            if metric_name == 'classes':
                df_table_criteria_style = df_table.loc[CRITERIA_ORDER].style
            else:
                df_table_criteria_style = df_table.loc[CRITERIA_ORDER].style.highlight_max(axis=0, props='bfseries: ;')
                
                if cfg.underline_worst:
                    df_table_criteria_style = df_table_criteria_style.highlight_min(axis=0, props='underline: ;')

            df_table_criteria_style.data = df_table.loc[CRITERIA_ORDER]
            df_table_criteria_style.data = df_table_criteria_style.data.rename(partial(format_criterium, train_samples=cfg.train_samples), axis=0)

            df_table_ceiling_style = df_table.loc[['full_data']].style
            df_table_ceiling_style.data = df_table_ceiling_style.data.rename(partial(format_criterium, train_samples=cfg.train_samples), axis=0)

            df_table_style = df_table_criteria_style.concat(df_table_ceiling_style)

            latex_str = df_table_style.to_latex(
                hrules=True,
                column_format='wr{3.7cm}' + '|' + len(DATASETS_ORDER) * 'wc{1.5cm}',
            )

            # final touches
            latex_str = re.sub(r'( \& matek)', r'sampling method \1', latex_str)
            latex_str = re.sub(r'(\\underline)([^0-9]*)([^ ]+)', r'\2\\underline{\3}', latex_str)
            latex_str = re.sub(r'(random)', r' \\midrule \1', latex_str)
            latex_str = re.sub(r'(cold paws)', r' \\midrule \1', latex_str)
            latex_str = re.sub(r'(full data)', r' \\midrule \\midrule \1', latex_str)

            file_name = f"{cfg.train_samples}_{ssl.replace('/', '_')}_{metric_name}.tex"
            with open(file_name, 'w') as f:
                f.write(latex_str)
                
            # buf=f"{cfg.train_samples}_{ssl.replace('/', '_')}_{metric_name}.tex", 

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
