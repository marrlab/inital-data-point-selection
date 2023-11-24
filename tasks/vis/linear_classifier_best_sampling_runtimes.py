
import re
import os
import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.utils.wandb import get_runs, get_best_criterium_filters
from src.vis.constants import DATASETS_ORDER, CRITERIA_ORDER, CRITERIA_ORDER_NO_RANDOM
from src.vis.helpers import mean_plus_minus_std, format_criterium
from src.utils.hydra import get_original_cwd_safe


@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_best_sampling_runtimes')
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

        def transform_runs_general(df):
            if len(df) == 0:
                return pd.DataFrame()

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['criterium'] = df.apply(
                lambda r: r['config']['kmeans']['criterium'], axis=1)
            df['sampling_time'] = df.apply(
                lambda r: r['config']['sampling_time'], axis=1)

            df = df.drop(['name', 'summary', 'config'], axis=1)
            df = df.dropna()

            return df

        def transform_runs_cold_paws(df):
            if len(df) == 0:
                return pd.DataFrame()

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['criterium'] = 'cold_paws'
            df['sampling_time'] = df.apply(
                lambda r: r['config']['sampling_time'], axis=1)

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

        # saving the dataframes
        df_general.to_pickle('df_general.pickle')
        df_cold_paws.to_pickle('df_cold_paws.pickle')
    else:
        print(f'loading checkpoint from: {cfg.checkpoint_dir}')
        df_general = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general.pickle'))
        df_cold_paws = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_cold_paws.pickle'))

    # merging dataframes
    df_general = pd.concat([df_general, df_cold_paws], ignore_index=True)
    df_general['criterium'] = pd.Categorical(df_general['criterium'], categories=CRITERIA_ORDER, ordered=True)
    df_general = df_general.sort_values(by='criterium')
    df_general['criterium'] = df_general['criterium'].apply(format_criterium)

    # creating the viz
    col_order = DATASETS_ORDER

    # creating the plot
    fig = sns.catplot(
        data=df_general,
        x='criterium',
        y='sampling_time',
        col='dataset', col_order=col_order,
        kind='bar',
        errorbar='sd',
        legend='full',
        log=True,
    )

    fig.savefig(f'{cfg.train_samples}.pdf')
    plt.clf()

    # creating the table
    df_temp = df_general.copy()

    # getting mean
    df_mean = df_temp.groupby(['dataset', 'criterium'], as_index=False).mean()
    df_mean = df_mean.pivot(index='dataset', columns='criterium')['sampling_time']
    df_mean = df_mean.loc[DATASETS_ORDER]
    df_mean = df_mean.T

    # getting std
    df_std = df_temp.groupby(['dataset', 'criterium'], as_index=False).std()
    df_std = df_std.pivot(index='dataset', columns='criterium')['sampling_time']
    df_std = df_std.loc[DATASETS_ORDER]
    df_std = df_std.T

    df_table = mean_plus_minus_std(df_mean=df_mean, df_std=df_std, decimal=1)

    latex_str = df_table.style.to_latex(
        hrules=True,
        column_format='wr{3.7cm}' + '|' + len(DATASETS_ORDER) * 'wc{1.5cm}'
    )

    # final touches
    latex_str = re.sub(r'( \& matek)', r'sampling method \1', latex_str)
    latex_str = re.sub(r'(random)', r' \\midrule \1', latex_str)
    latex_str = re.sub(r'(cold paws)', r' \\midrule \1', latex_str)
    
    file_name = f'{cfg.train_samples}_sampling-time.tex'
    with open(file_name, 'w') as f:
        f.write(latex_str)

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
