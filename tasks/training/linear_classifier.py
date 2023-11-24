
import os
import uuid
import time
import hydra
import torch
import copy
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from collections import defaultdict
import torchvision
import lightning.pytorch as pl
from src.datasets.datasets import get_dataset_class_by_name, ImageFeaturesDataset
from src.models.classifiers import get_ssl_preprocess, Classifier 
from src.models.scan import get_classifier_from_scan
from src.datasets.subsets import get_n_clustered
from src.models.training import train_image_classifier
from src.models.build_data_path import get_features_path, get_scan_features_path, get_precomputed_features_folder_path
from src.utils.wandb import init_run, cast_dict_to_int


@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier')
def main(cfg: DictConfig):
    # # debug
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return

    assert cfg.use_scan_weights == False
    assert cfg.training.learning_rate is None
    assert cfg.training.model_save_path is None
    assert cfg.training.train_samples is not None

    # we don't want this due to different sampling each time
    if cfg.training.seed is not None:
        print(f'setting random seed: {cfg.training.seed}')
        pl.seed_everything(cfg.training.seed)

    features_path = None
    if cfg.use_scan_weights:
        features_path = get_scan_features_path(cfg, absolute=False)
    else:
        features_path = get_features_path(cfg, absolute=False)
    print(f'using features stored at the path: {features_path}')

    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class(
        'train', 
        features_path=features_path,
        load_images=False
    )
    # val_dataset = dataset_class('val', preprocess=preprocess)
    # test_dataset = dataset_class('test', preprocess=preprocess)

    # feature scaling
    if cfg.features.scaling is None:
        pass
    elif cfg.features.scaling == 'standard':
        train_dataset.standard_scale_features()
    elif cfg.features.scaling == 'min_max':
        train_dataset.min_max_scale_features()
    else:
        raise ValueError(f'unknown feature scaling: {cfg.feature.scaling}')

    # clustering
    train_subset = None
    if cfg.training.train_samples is None:
        print('using all the train samples')
        train_subset = train_dataset
    else:
        print(f'picking {cfg.training.train_samples} train samples')
        start = time.time()

        train_subset = get_n_clustered(train_dataset,
                                    n_samples=cfg.training.train_samples, n_clusters=cfg.kmeans.clusters,
                                    mode=cfg.kmeans.mode, criterium=cfg.kmeans.criterium)

        cfg.sampling_time = time.time() - start
        cfg.names = [n for n in train_subset.images_data['names']]

    train_subset.reassign_classes()
    label_counts = defaultdict(int)
    for label in train_subset.images_data['labels']:
        label_counts[label] += 1
    cfg.labels = train_subset.get_number_of_labels()
    cfg.labels_text = train_subset.labels_text
    cfg.label_counts = cast_dict_to_int(label_counts)
    cfg.label_to_class_mapping = cast_dict_to_int(
        train_subset.label_to_class_mapping)
    cfg.class_to_label_mapping = cast_dict_to_int(
        train_subset.class_to_label_mapping)
    cfg.classes = train_subset.get_number_of_classes()

    # val_dataset.match_classes(train_subset)
    # test_dataset.match_classes(train_subset)

    num_classes = train_subset.get_number_of_classes()

    # initializing features detaset
    path = get_precomputed_features_folder_path(cfg, absolute=True)
    train_features_dataset = ImageFeaturesDataset('train', path)
    train_features_dataset.load(train_subset.images_data['names'])
    train_features_dataset.match_classes(train_subset)

    test_features_dataset = ImageFeaturesDataset('test', path)
    test_features_dataset.load(None)
    test_features_dataset.match_classes(train_subset)

    model_state_dicts = []
    for learning_rate in cfg.soup.repetitions * list(cfg.soup.learning_rates):
        print(f'training a linear head: learning_rate={learning_rate}')

        # initializing the model
        model = Classifier(
            backbone=torch.nn.Identity(), 
            backbone_output_dim=train_dataset.get_feature_dim(),
            freeze_backbone=True,
            num_classes=num_classes 
        )
    
        cfg.training.learning_rate = learning_rate
        cfg.training.model_save_path = f'{uuid.uuid4()}.ckpt'

        init_run(cfg)

        start = time.time()
        train_image_classifier(
            model=model, 
            train_dataset=train_features_dataset, 
            val_dataset=None,
            test_dataset=test_features_dataset, 
            cfg=cfg,
            log_images=False,
        )
        wandb.config.training_time = time.time() - start

        wandb.finish()

        # saving the state dict
        model_state_dicts.append(model.state_dict())

    # creating the soup
    state_dict = None
    for model_state_dict in model_state_dicts:
        # initializing the state dict
        if state_dict is None:
            state_dict = copy.deepcopy(model_state_dict)
            for key in model_state_dict:
                state_dict[key] *= 0

        # averiging
        for key in model_state_dict:
            state_dict[key] += model_state_dict[key] / len(model_state_dicts)

    model_soup = Classifier(
        backbone=torch.nn.Identity(), 
        backbone_output_dim=train_dataset.get_feature_dim(),
        freeze_backbone=True,
        num_classes=num_classes 
    )
    model_soup.load_state_dict(state_dict)

    # evaluating the soup
    cfg.wandb.project = f'{cfg.wandb.project}-soup'

    init_run(cfg)

    start = time.time()
    train_image_classifier(
        model=model_soup, 
        train_dataset=None, 
        val_dataset=None,
        test_dataset=test_features_dataset, 
        cfg=cfg,
        log_images=False,
    )
    wandb.config.training_time = time.time() - start

    wandb.finish()

if __name__ == '__main__':
    main()
