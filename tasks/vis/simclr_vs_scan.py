

import os
import hydra
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.hydra import get_original_cwd_safe
from src.utils.wandb import get_runs, dict_to_filters
from src.models.build_data_path import get_vis_folder_path


@hydra.main(version_base=None, config_path='../../conf', config_name='simclr_vs_scan')
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

    # general
    filters = dict_to_filters({
        'training': {
            'train_samples': None,
            'weights': {
                'type': 'simclr',
                'version': 'v1'
            }
        },
    })

    def transform_runs(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)
        
        df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
        df['scan'] = df.apply(lambda r: r['config']['use_scan_weights'], axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df = get_runs('badge-sampling', filters=filters)
    df = transform_runs(df)
    print(f'total runs: {len(df)}')

    # saving the dataframes
    df.to_pickle('df.pickle')

    # creating the plots
    for metric_name in required_metrics_name:
        fig = sns.lineplot(data=df, x='dataset', y=metric_name, hue='scan', marker='o', linestyle='').get_figure()
        fig.savefig(f'{metric_name}.pdf')
        fig.clf()


if __name__ == '__main__':
    main()
