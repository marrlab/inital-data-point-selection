
import hydra
import copy
import wandb
from omegaconf import DictConfig
import lightning.pytorch as pl
from src.datasets.datasets import get_dataset_class_by_name
# TODO
from src.models.classifiers import \
    get_classifier_imagenet, get_classifier_imagenet_preprocess_only, get_classifier_from_simclr, get_classifier_simclr_preprocess_only
from src.datasets.subsets import get_n_kmeans
from src.models.training import train_image_classifier
from src.utils.wandb import init_run

@hydra.main(version_base=None, config_path='../../conf', config_name='badge_sampling')
def main(cfg: DictConfig):
    init_run(cfg)

    pl.seed_everything(cfg.training.seed)

    preprocess = None
    if cfg.training.weights.type == 'imagenet':
        preprocess = get_classifier_imagenet_preprocess_only(
            cfg.training.architecture)
    elif cfg.training.weights.type == 'simclr':
        preprocess = get_classifier_simclr_preprocess_only(
            cfg.training.input_size)
    else:
        raise ValueError(f'unknown weights type: {cfg.training.weights.type}')

    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    train_dataset = dataset_class('train', preprocess=preprocess, features_path=cfg.features.path)
    val_dataset = dataset_class('test', preprocess=preprocess)

    # feature scaling
    if cfg.features.scaling is None:
        pass
    elif cfg.features.scaling == 'standard':
        train_dataset.standard_scale_features()
    elif cfg.features.scaling == 'min_max':
        train_dataset.min_max_scale_features()
    else:
        raise ValueError(f'unknown feature scaling: {cfg.feature_scaling}')

    train_subset = get_n_kmeans(train_dataset, cfg.training.train_samples,
                                mode=cfg.kmeans.mode, criterium=cfg.kmeans.criterium)
    # TODO
    train_subset.relabel()
    wandb.config.labels = len(train_subset.labels)
    wandb.config.labels_text = train_subset.labels_text

    val_subset = copy.deepcopy(val_dataset)
    val_subset.match_labels_and_filter(train_subset)

    num_classes = len(train_subset.labels)

    model = None
    if cfg.training.weights.type == 'imagenet':
        # TODO: add weight freezing option
        model, _ = get_classifier_imagenet(cfg.training.architecture, num_classes)
    elif cfg.training.weights.type == 'simclr':
        # TODO
        model = get_classifier_from_simclr(preprocess, cfg, num_classes)
    else:
        raise ValueError(f'unknown weights type: {cfg.training.weights.type}')

    train_image_classifier(model, train_subset, val_subset, cfg)

if __name__ == '__main__':
    main()
