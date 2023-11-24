
import hydra
import copy
import wandb
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import torchvision
import lightning.pytorch as pl
from src.datasets.datasets import get_dataset_class_by_name
from src.models.classifiers import get_ssl_preprocess, get_classifier_from_ssl, get_ssl_transform
from src.models.scan import get_classifier_from_scan
from src.datasets.subsets import get_n_clustered
from src.models.training import train_image_classifier
from src.models.build_data_path import get_features_path, get_scan_features_path
from src.utils.wandb import init_run, cast_dict_to_int


@hydra.main(version_base=None, config_path='../../conf', config_name='badge_sampling')
def main(cfg: DictConfig):
    # # debug
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return

    init_run(cfg)

    # we don't want this due to different sampling each time
    if cfg.training.seed is not None:
        print(f'setting random seed: {cfg.training.seed}')
        pl.seed_everything(cfg.training.seed)

    preprocess = get_ssl_preprocess(cfg)
    transform = get_ssl_transform(cfg)    

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
        transform=transform, 
        preprocess=torchvision.transforms.Resize(cfg.dataset.input_size, antialias=True)
    )
    # val_dataset = dataset_class('val', preprocess=preprocess)
    test_dataset = dataset_class('test', preprocess=preprocess)

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
        train_subset = get_n_clustered(train_dataset,
                                    n_samples=cfg.training.train_samples, n_clusters=cfg.kmeans.clusters,
                                    mode=cfg.kmeans.mode, criterium=cfg.kmeans.criterium)
        wandb.config.names = [n for n in train_subset.images_data['names']]

    train_subset.reassign_classes()
    label_counts = defaultdict(int)
    for label in train_subset.images_data['labels']:
        label_counts[label] += 1
    wandb.config.labels = train_subset.get_number_of_labels()
    wandb.config.labels_text = train_subset.labels_text
    wandb.config.label_counts = cast_dict_to_int(label_counts)
    wandb.config.label_to_class_mapping = cast_dict_to_int(
        train_subset.label_to_class_mapping)
    wandb.config.class_to_label_mapping = cast_dict_to_int(
        train_subset.class_to_label_mapping)
    wandb.config.classes = train_subset.get_number_of_classes()

    # val_dataset.match_classes(train_subset)
    test_dataset.match_classes(train_subset)

    num_classes = train_subset.get_number_of_classes()

    model = None
    if cfg.use_scan_weights:
        print('using scan weights')
        model = get_classifier_from_scan(cfg, num_classes)
    else:
        print('using ssl weights')
        model = get_classifier_from_ssl(cfg, train_dataset.get_feature_dim(), num_classes)
    train_image_classifier(model, train_subset, None, test_dataset, cfg)

    wandb.finish()


if __name__ == '__main__':
    main()
