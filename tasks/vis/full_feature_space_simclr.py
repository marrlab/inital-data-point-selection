

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


@hydra.main(version_base=None, config_path='../../conf', config_name='full_feature_space_simclr')
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

    filters_random = dict_to_filters({
        'kmeans': {
            'clusters': 1,
            'criterium': 'random'
        }
    })
    filters_random = {**filters_general, **filters_random}

    filters_fps = dict_to_filters({
        'kmeans': {
            'clusters': 1,
            'criterium': 'fps'
        },
        'features': {
            'scaling': 'standard'
        }
    })
    filters_fps = {**filters_general, **filters_fps}

    filters_ceiling = dict_to_filters({
        'training': {
            'train_samples': None,
            'weights': {
                'type': 'simclr',
                'version': 'v1'
            }
        },
    })

    # dropout uncertainty
    filters_dropout = dict_to_filters({
        'training': {
            'weights': {
                'type': 'simclr',
                'version': 'v1'
            }
        },
        'scan': {
            'dropout': 0.5
        }
    })
    filters_dropout = {**filters_general, **filters_dropout}

    # fps cosine
    filters_fps_cosine = dict_to_filters({
        'use_scan_weights': False
    })
    filters_fps_cosine = {**filters_general, **filters_fps_cosine}

    # fps l1
    filters_fps_l1 = dict_to_filters({
        'use_scan_weights': False
    })
    filters_fps_l1 = {**filters_general, **filters_fps_l1}

    # cold paws
    filters_cold_paws = dict_to_filters({
        'training': {
            'weights': {
                'type': 'simclr',
                'version': 'v1'
            }
        },
    })
<<<<<<< HEAD
    filters_cold_paws = {**filters_cold_paws, **filters_dropout}
=======
    filters_cold_paws = {**filters_general, **filters_cold_paws}
>>>>>>> 7019fb84c2f2c2ca6374da3f4c93b4fb901f2a54

    # transforms
    def transform_runs(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)
        
        df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
        df['scan'] = df.apply(lambda r: r['config'].get('use_scan_weights', None), axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df_random = get_runs('badge-sampling', filters=filters_random)
    df_random = transform_runs(df_random)
    print(f'total runs (random): {len(df_random)}')
    print(df_random)

    df_fps = get_runs('badge-sampling', filters=filters_fps)
    df_fps = transform_runs(df_fps)
    print(f'total runs (fps): {len(df_fps)}')
    print(df_fps)

    df_ceiling = get_runs('badge-sampling', filters=filters_ceiling)
    df_ceiling = transform_runs(df_ceiling)
    print(f'total runs (ceiling): {len(df_ceiling)}')
    print(df_ceiling)

    df_dropout = get_runs('monte-carlo', filters=filters_dropout)
    df_dropout = transform_runs(df_dropout)
    print(f'total runs (dropout): {len(df_dropout)}')
    print(df_dropout)

    df_fps_cosine = get_runs('simclr-fps-cosine', filters=filters_fps_cosine)
    df_fps_cosine = transform_runs(df_fps_cosine)
    print(f'total runs (fps cosine): {len(df_fps_cosine)}')
    print(df_fps_cosine)
    
    df_fps_l1 = get_runs('simclr-fps-l1', filters=filters_fps_l1)
    df_fps_l1 = transform_runs(df_fps_l1)
    print(f'total runs (fps l1): {len(df_fps_l1)}')
    print(df_fps_l1)

<<<<<<< HEAD
    df_cold_paws = get_runs('simclr-cold-paws', filters=filters_cold_paws)
=======
    df_cold_paws = get_runs('cold-paws', filters=filters_cold_paws)
>>>>>>> 7019fb84c2f2c2ca6374da3f4c93b4fb901f2a54
    df_cold_paws = transform_runs(df_cold_paws)
    print(f'total runs (cold paws): {len(df_cold_paws)}')
    print(df_cold_paws)

    # concatenating
    df_random['method'] = df_random.apply(lambda r: 'scan-random' if r['scan'] else 'simclr-random', axis=1)
    df_fps['method'] = df_fps.apply(lambda r: 'scan-fps-l2' if r['scan'] else 'simclr-fps-l2', axis=1) 
    df_dropout['method'] = 'scan-dropout'
    df_fps_cosine['method'] = 'simclr-fps-cosine'
    df_fps_l1['method'] = 'simclr-fps-l1'
    df_cold_paws['method'] = 'simclr-cold-paws'
    df = pd.concat([df_random, df_fps, df_dropout, df_fps_cosine, df_fps_l1, df_cold_paws])

    # creating the plots
    for metric_name in required_metrics_name:
        fig = sns.pointplot(
            data=df,
            x='dataset', y=metric_name, hue='method',
            join=False, dodge=0.5, errorbar='sd',
            order=['matek', 'isic', 'retinopathy', 'jurkat', 'cifar10'],
            hue_order=['simclr-random', 'simclr-fps-l2', 'simclr-fps-l1', 'simclr-fps-cosine', 'simclr-cold-paws', 'scan-random', 'scan-fps-l2', 'scan-dropout'],
        ).get_figure()
        fig.savefig(f'{metric_name}.pdf')
        fig.clf()

    print(f'everything saved to: {os.getcwd()}')


if __name__ == '__main__':
    main()
