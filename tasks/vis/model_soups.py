import os
import copy
import hydra
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from src.datasets.datasets import get_dataset_class_by_name
from src.models.classifiers import get_ssl_preprocess, get_classifier_from_ssl
from src.datasets.subsets import get_by_names
from src.models.build_data_path import get_features_path
from src.utils.wandb import get_runs, dict_to_filters
from src.utils.hydra import get_original_cwd_safe
from src.models.lightning_modules import ImageClassifierLightningModule
from src.utils.utils import get_cpu_count, get_the_best_accelerator, to_best_available_device


@hydra.main(version_base=None, config_path='../../conf', config_name='model_soups')
def main(cfg: DictConfig):
    # # debug
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return

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

    # retrieving data from wandb
    filters_general = dict_to_filters({
        'dataset': dict(cfg.dataset),
        'training': {
            'seed': cfg.training.seed,
            'train_samples': cfg.training.train_samples,
            'oversample': cfg.training.oversample,
            'weights': dict(cfg.training.weights),
        },
        'kmeans': dict(cfg.kmeans)
    })
    df_general = get_runs('model-soups', filters=filters_general)
    print(f'total runs: {len(df_general)}')

    def transform_runs_general(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)

        df['names'] = df.apply(
            lambda r: r['config']['names'], axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df_general = transform_runs_general(df_general)

    # checking if all runs are actually the same
    names = None
    for n in df_general['names']:
        n = sorted(n)

        if names is None:
            names = n

        if n != names:
            raise ValueError('sample names are not the same for all the runs')

    df_general = df_general.drop(['names'], axis=1)
    df_general['model'] = 'model_' + df_general.index.astype(str)

    print(df_general)

    # loading the train dataset to recreate classes observed
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class(
        'train',
        features_path=get_features_path(cfg, absolute=False),
    )
    train_subset = get_by_names(train_dataset, names)
    train_subset.reassign_classes()

    # loading the test dataset
    preprocess = get_ssl_preprocess(cfg)
    test_dataset = dataset_class('test', preprocess=preprocess)
    test_dataset.match_classes(train_subset)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=get_cpu_count()
    )

    # loading all the models
    model_state_dicts = []
    for path in cfg.model_paths:
        model_uninitialized = get_classifier_from_ssl(
            cfg,
            train_dataset.get_feature_dim(),
            train_subset.get_number_of_classes()
        )

        model = ImageClassifierLightningModule.load_from_checkpoint(
            os.path.join(get_original_cwd_safe(), path),
            model=model_uninitialized,
            num_labels=train_subset.get_number_of_labels(),
            num_classes=train_subset.get_number_of_classes(),
            label_to_class_mapping=train_subset.label_to_class_mapping.copy(),
            class_to_label_mapping=train_subset.class_to_label_mapping.copy(),
            cfg=cfg
        )

        model_state_dicts.append(model.model.state_dict())

    assert len(df_general) == len(model_state_dicts), 'number of runs and models differs'

    # creating the soup
    state_dict = None
    for model_state_dict in model_state_dicts:
        if state_dict is None:
            state_dict = copy.deepcopy(model_state_dict)
            for key in model_state_dict:
                if not key.endswith('num_batches_tracked'):
                    state_dict[key] *= 0

        for key in model_state_dict:
            if not key.endswith('num_batches_tracked'):
                state_dict[key] += model_state_dict[key] / len(model_state_dicts)

    model_soup_uninitialized = get_classifier_from_ssl(
        cfg,
        train_dataset.get_feature_dim(),
        train_subset.get_number_of_classes()
    )
    model_soup_uninitialized.load_state_dict(state_dict)
    model_soup = ImageClassifierLightningModule(
        model=model_soup_uninitialized,
        num_labels=train_subset.get_number_of_labels(),
        num_classes=train_subset.get_number_of_classes(),
        label_to_class_mapping=train_subset.label_to_class_mapping.copy(),
        class_to_label_mapping=train_subset.class_to_label_mapping.copy(),
        cfg=cfg
    )

    # inference for all the models
    trainer = pl.Trainer(
        accelerator=get_the_best_accelerator(),
        devices=1,
    )
    result = trainer.test(model_soup, dataloaders=test_data_loader)[0]

    # putting everything in dataframe
    d = {}
    d['model'] = 'model_soup'
    for i in range(len(required_metrics)):
        d[required_metrics_name[i]] = result[required_metrics[i]]
    
    print(d)

    df_general = pd.concat([df_general, pd.DataFrame([d])], ignore_index=True)
    print(df_general)

    # saving the dataframe
    df_general.to_pickle('df_general.pickle')

    # creating the vis
    palette = ['orange' if x == 'model_soup' else 'grey' for x in df_general['model']]
    for metric_name in required_metrics_name:
        fig = sns.catplot(df_general, x='model', y=metric_name, kind='bar', palette=palette)
        fig.savefig(f'{cfg.dataset.name}_{metric_name}.pdf')
        plt.clf()

if __name__ == '__main__':
    main()
