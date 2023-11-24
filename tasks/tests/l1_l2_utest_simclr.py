

import os
import hydra
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from scipy.stats import mannwhitneyu
from src.utils.utils import latex_to_pdf
from src.utils.wandb import get_runs, dict_to_filters


@hydra.main(version_base=None, config_path='../../conf', config_name='l1_l2_utest_simclr')
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

    # fps l1
    filters_fps_l1 = dict_to_filters({
        'use_scan_weights': False
    })
    filters_fps_l1 = {**filters_general, **filters_fps_l1}

    # transforms
    def transform_runs(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)
        
        df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
        df['scan'] = df.apply(lambda r: r['config'].get('use_scan_weights', None), axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df_fps_l2 = get_runs('badge-sampling', filters=filters_fps_l2)
    df_fps_l2 = transform_runs(df_fps_l2)
    print(f'total runs (fps l2): {len(df_fps_l2)}')
    print(df_fps_l2)
    
    df_fps_l1 = get_runs('simclr-fps-l1', filters=filters_fps_l1)
    df_fps_l1 = transform_runs(df_fps_l1)
    print(f'total runs (fps l1): {len(df_fps_l1)}')
    print(df_fps_l1)

    # creating the plots
    d = {
        'dataset': [],
        'metric': [],
        'p-value': [],
    }
    for metric_name in required_metrics_name:
        for dataset in ('matek', 'isic', 'retinopathy', 'jurkat', 'cifar10'):
            d['dataset'].append(dataset)
            d['metric'].append(metric_name)

            x = df_fps_l2[df_fps_l2['dataset'] == dataset][metric_name]
            y = df_fps_l1[df_fps_l1['dataset'] == dataset][metric_name]
            _, p_value = mannwhitneyu(x, y, alternative='greater')
            d['p-value'].append(p_value)

    df = pd.DataFrame(d)
    df.to_pickle('df.pickle')

    print('results:')
    print(df)

    def round_to_hundreds(value):
        return round(value, 2)

    def confident_to_bold(value):
        def two_decimal_string(value):
            return '{:.2f}'.format(value)

        if value <= 0.05:
            return '\\textbf{' + two_decimal_string(value) + '}'
        else:
            return two_decimal_string(value)

    df = df.pivot(index='dataset', columns='metric')
    df = df.applymap(round_to_hundreds)
    df = df.applymap(confident_to_bold)

    df.to_latex(buf='df.tex')
    latex_to_pdf('df.tex', 'df.pdf')

    print(f'everything saved to: {os.getcwd()}')


if __name__ == '__main__':
    main()
