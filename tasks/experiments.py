
import copy
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets.datasets import ImageDataset, MatekDataset
from models.classifiers import \
    get_classifier_imagenet, get_classifier_imagenet_preprocess_only
from datasets.subsets import get_n_random, get_n_kmeans
from tasks.training import train_image_classifier


# wandb.config requirements:
# .epochs
# .learning_rate
# .batch_size
# .architecture
# .dataset
# .train_samples
# .val_samples
def random_baseline():
    assert wandb.config.dataset in ('matek')

    preprocess = get_classifier_imagenet_preprocess_only(
        wandb.config.architecture)

    train_dataset, val_dataset = None, None
    if wandb.config.dataset == 'matek':
        train_dataset = MatekDataset('train', preprocess=preprocess)
        val_dataset = MatekDataset('test', preprocess=preprocess)

    train_subset = get_n_random(train_dataset, wandb.config.train_samples)
    train_subset.relabel()

    val_subset = copy.deepcopy(val_dataset)
    val_subset.match_labels_and_filter(train_subset)
    val_subset = get_n_random(val_subset, wandb.config.val_samples)

    num_classes = len(train_subset.labels)
    model, _ = get_classifier_imagenet(wandb.config.architecture, num_classes)

    train_image_classifier(model, train_subset, val_subset)


# wandb.config requirements:
# .epochs
# .learning_rate
# .batch_size
# .architecture
# .dataset
# .train_samples
# .val_samples
# .features_path
# .mode
# .criterium
def badge_sampling():
    assert wandb.config.dataset in ('matek')
    assert 'feature_scaling' not in wandb.config or wandb.config.feature_scaling in ('standard', 'min_max')

    preprocess = get_classifier_imagenet_preprocess_only(
        wandb.config.architecture)

    train_dataset, val_dataset = None, None
    if wandb.config.dataset == 'matek':
        train_dataset = MatekDataset(
            'train', preprocess=preprocess, features_path=wandb.config.features_path)
        val_dataset = MatekDataset('test', preprocess=preprocess)

    if 'feature_scaling' in wandb.config:
        if wandb.config.feature_scaling == 'standard':
            train_dataset.standard_scale_features()
        elif wandb.config.feature_scaling == 'min_max':
            train_dataset.min_max_scale_features()

    train_subset = get_n_kmeans(train_dataset, wandb.config.train_samples,
                                mode=wandb.config.mode, criterium=wandb.config.criterium)
    train_subset.relabel()

    val_subset = copy.deepcopy(val_dataset)
    val_subset.match_labels_and_filter(train_subset)
    val_subset = get_n_random(val_subset, wandb.config.val_samples)

    num_classes = len(train_subset.labels)
    model, _ = get_classifier_imagenet(wandb.config.architecture, num_classes)

    train_image_classifier(model, train_subset, val_subset)


def subsetting_methods_performance(dataset: ImageDataset, runs: int, n: int) -> pd.DataFrame:
    assert dataset.features_path is not None

    ds = []

    def generate_summary(method_name: str, subset: ImageDataset) -> dict:
        d = {}
        d['method'] = method_name
        for l in dataset.labels:
            d[dataset.labels_text[l]] = np.sum(
                np.array(subset.images_data['labels']) == l)

        return d

    # true distribution
    ds.append(generate_summary('true', dataset))

    for _ in tqdm(range(runs)):
        ds.append(generate_summary('random', get_n_random(dataset, n)))
        ds.append(generate_summary('badge_kmeans++_closest',
                  get_n_kmeans(dataset, n, mode='kmeans++', criterium='closest')))
        ds.append(generate_summary('badge_kmeans++_furthest',
                  get_n_kmeans(dataset, n, mode='kmeans++', criterium='furthest')))
        ds.append(generate_summary('badge_kmeans_closest', get_n_kmeans(
            dataset, n, mode='kmeans', criterium='closest')))
        ds.append(generate_summary('badge_kmeans_furthest', get_n_kmeans(
            dataset, n, mode='kmeans', criterium='furthest')))

    return pd.DataFrame(ds)
