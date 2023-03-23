
import sys
import copy
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.datasets import ImageDataset, MatekDataset
from src.datasets.datasets import get_dataset_class_by_name
from src.models.classifiers import \
    get_classifier_imagenet, get_classifier_imagenet_preprocess_only
from src.datasets.subsets import get_n_random, get_n_kmeans
from src.models.training import train_image_classifier
from sklearn.cluster import kmeans_plusplus, KMeans
from src.utils.utils import load_yaml_as_dict, load_yaml_as_obj




def random_baseline():
    preprocess = get_classifier_imagenet_preprocess_only(
        wandb.config.architecture)

    dataset_class = get_dataset_class_by_name(wandb.config.dataset)
    train_dataset = dataset_class('train', preprocess=preprocess)
    val_dataset = dataset_class('test', preprocess=preprocess)

    train_subset = get_n_random(train_dataset, wandb.config.train_samples)
    train_subset.relabel()
    wandb.config.labels = len(train_subset.labels)
    wandb.config.labels_text = train_subset.labels_text

    val_subset = copy.deepcopy(val_dataset)
    val_subset.match_labels_and_filter(train_subset)

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
    assert 'feature_scaling' not in wandb.config or wandb.config.feature_scaling in ('standard', 'min_max')

    preprocess = get_classifier_imagenet_preprocess_only(
        wandb.config.architecture)

    dataset_class = get_dataset_class_by_name(wandb.config.dataset)
    train_dataset = dataset_class('train', preprocess=preprocess, features_path=wandb.config.features_path)
    val_dataset = dataset_class('test', preprocess=preprocess)

    if 'feature_scaling' in wandb.config:
        if wandb.config.feature_scaling == 'standard':
            train_dataset.standard_scale_features()
        elif wandb.config.feature_scaling == 'min_max':
            train_dataset.min_max_scale_features()

    train_subset = get_n_kmeans(train_dataset, wandb.config.train_samples,
                                mode=wandb.config.mode, criterium=wandb.config.criterium)
    train_subset.relabel()
    wandb.config.labels = len(train_subset.labels)
    wandb.config.labels_text = train_subset.labels_text

    val_subset = copy.deepcopy(val_dataset)
    val_subset.match_labels_and_filter(train_subset)

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


def cluster_data_points_analysis(dataset: ImageDataset, clusters: int) -> pd.DataFrame:
    features = []
    labels = []
    label_names = []
    
    for data_point in dataset:
        features.append(data_point['feature'])
        labels.append(data_point['label'])
        label_names.append(data_point['name'])

    features = np.array(features)
    labels = np.array(labels)
    label_names = np.array(label_names)

    kmeans = KMeans(n_clusters=clusters).fit(features)

    # per feature
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_[cluster_labels]
    distances_to_cluster_centers = np.linalg.norm(features - cluster_centers, axis=1)

    return pd.DataFrame({
        'label': labels,
        'label_name': label_names,
        'cluster_label': cluster_labels,
        'distance_to_cluster_center': distances_to_cluster_centers
    })

if __name__ == '__main__':
    task_name = sys.argv[1]
    config_path = sys.argv[2] 

    config = load_yaml_as_dict(config_path)

    if task_name == 'random_baseline':
        wandb.init(project='random-baseline', config=config)
        random_baseline()
    elif task_name == 'badge_sampling':
        wandb.init(project='badge-sampling', config=config)
        badge_sampling()
    else:
        raise ValueError(f'unknown task name: {task_name}')

    # wandb wrapping-up
    wandb.finish()
    