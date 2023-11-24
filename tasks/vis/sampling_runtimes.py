

import os
import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs, dict_to_filters
from src.models.build_data_path import get_vis_folder_path


@hydra.main(version_base=None, config_path='../../conf', config_name='sampling_runtimes')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')

    # filters
    filters_general = dict_to_filters({
        'training': {
            'train_samples': 100,
            'oversample': True,
            'weights': {
                'freeze': True,
            },
        },
        'features': {
            'scaling': 'standard'
        },
        'kmeans': {
            'mode': 'kmeans'
        },
    })
    filters_general['config.use_scan_weights'] = {'$in': [None, False]}
    filters_general['config.training.weights.version'] = {'$ne': None}

    def transform_runs_general(df):
        if len(df) == 0:
            return pd.DataFrame()

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
        df['sampling_time'] = df.apply(
            lambda r: r['config']['sampling_time'],
            axis=1
        ) 

        df = df.drop(['name', 'summary', 'config'], axis=1)
        df = df.dropna()

        return df

    print('fetching all general runs')
    df_general = get_runs('linear-classifier-soup', filters=filters_general)
    df_general = transform_runs_general(df_general)
    print(f'total runs: {len(df_general)}')
    print(df_general.groupby(['ssl']).count())
    print()

    # saving the dataframes
    df_general.to_pickle('df_general.pickle')

    # creating the plots
    for ssl in df_general['ssl'].unique():
        df_ssl = df_general[df_general['ssl'] == ssl]

        fig = sns.catplot(
            data=df_ssl, 
            x='clusters', y='sampling_time', 
            col='dataset', hue='criterium',
            kind='bar',
            col_order=['matek', 'isic', 'retinopathy', 'jurkat', 'cifar10']
        )

        fig.savefig(f"{ssl.replace('/', '-')}.pdf")
        plt.clf()

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
