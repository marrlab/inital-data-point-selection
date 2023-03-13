
import os
import sys
import umap
import umap.plot
import wandb
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
from utils.utils import load_dataframes
from scipy.stats import entropy
from utils.types import Result
from utils.utils import load_yaml_as_obj
from datasets.datasets import get_dataset_class_by_name
from utils.utils import result_to_dataframe, get_runs
from copy import deepcopy


def subsetting_methods_performance_preprocessing(result: Result) -> pd.DataFrame:
    df = result_to_dataframe(result)
    df = df.groupby(['method']).mean()
    df = df.div(df.sum(axis=1), axis=0)
    
    def rename_method(method: str) -> str:
        if method.startswith('badge_'):
            method = method[6:]
            method = method.replace('_', ' ')

        return method

    df = df[df.index != 'true']
    df.index = df.index.map(rename_method)

    return df


def subsetting_methods_performance_heatmap(result: Result, path: str = None):
    df = subsetting_methods_performance_preprocessing(result)
    df *= 100

    _, ax = plt.subplots(figsize=(15,5))

    sns.heatmap(df, annot=True, annot_kws={'fontsize': 13}, fmt='.1f', linewidths=.5, cmap='gray_r', ax=ax)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def subsetting_methods_performance_entropy(result: Result, path: str):
    df = subsetting_methods_performance_preprocessing(result)
    labels_count = len(df.columns)

    df['entropy'] = df.apply(lambda x: entropy(x), axis=1)
    df = df[['entropy']]
    df['entropy'] = df['entropy'].round(decimals=2)

    # normalized entropy
    # print(df['entropy'] / entropy(labels_count*[1 / labels_count]))

    dfi.export(df, path)


def cluster_data_points_analysis(result_path: str, path: str = None):
    df = load_dataframes([result_path], contains_index=False)[0]

    cluster_label_counts = df.groupby(['cluster_label', 'label'])['cluster_label'].count().to_dict()
    cluster_label_min_distances = df.groupby(['cluster_label', 'label'])['distance_to_cluster_center'].min().to_dict()

    clusters_count = len(df['cluster_label'].unique())
    labels_count = len(df['label'].unique())

    cluster_min_distance_label = []
    cluster_majority_label = []
    cluster_labels_entropy = []
    for c in range(clusters_count):
        labels_distribution = np.array([
            cluster_label_counts[(c, l)] if (c, l) in cluster_label_counts else 0
            for l in range(labels_count)
        ])
        labels_distribution = labels_distribution / np.sum(labels_distribution)
        
        labels_distances = np.array([
            cluster_label_min_distances[(c, l)] if (c, l) in cluster_label_min_distances else np.inf
            for l in range(labels_count)
        ])

        cluster_min_distance_label.append(np.argmin(labels_distances))
        cluster_majority_label.append(np.argmax(labels_distribution))
        cluster_labels_entropy.append(entropy(labels_distribution))

    result_df = {
        'indicator': ['label_with_min_distance', 'majority_label', 'label_distribution_entropy'],
    }

    for c in range(clusters_count):
        result_df[f'cluster_{c}'] = [
            cluster_min_distance_label[c],
            cluster_majority_label[c],
            round(cluster_labels_entropy[c], 2),
        ]

    result_df = pd.DataFrame(result_df).transpose()
    result_df.columns = result_df.iloc[0]
    result_df = result_df[1:]
    result_df = result_df.astype({
        'label_with_min_distance': int,
        'majority_label': int 
    })

    dfi.export(result_df, path)


def classification_metrics(dataset_name: str, path_mean: str=None, path_std: str=None, path_mean_std: str=None):
    wandb.login(key='a29d7c338a594e427f18a0f1502e5a8f36e9adfb')
    api = wandb.Api()

    # defined fields
    required_metrics = [
        'val_accuracy_epoch_end_max',
        'val_f1_macro_epoch_end_max',
        'val_balanced_accuracy_max',
        'val_matthews_corrcoef_max',
        'val_cohen_kappa_score_max'
    ]
    required_metrics_name = [
        'accuracy',
        'f1 macro',
        'balanced accuracy',
        'matthews corrcoef',
        'cohen kappa score'
    ]

    # random baseline
    def filter_run_random(row):
        if row['config']['dataset'] != dataset_name:
            return False

        if 'epoch' not in row['summary'] or row['summary']['epoch'] != 29:
            return False
        
        if any(rm not in row['summary'] for rm in required_metrics):
            return False
        
        return True
    
    def transform_runs_random(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)

        df['classes'] = df.apply(lambda r: r['config']['num_classes'], axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df_random = get_runs('random-baseline')
    filtered_rows = df_random.apply(filter_run_random, axis=1)
    df_random = df_random[filtered_rows]
    df_random = transform_runs_random(df_random)
    df_random['method'] = 'random'
    
    # badge sampling
    def filter_run_badge(row):
        if row['config']['dataset'] != dataset_name:
            return False

        if 'feature_scaling' not in row['config'] or row['config']['feature_scaling'] != 'standard':
            return False

        if 'epoch' not in row['summary'] or row['summary']['epoch'] != 29:
            return False
        
        if any(rm not in row['summary'] for rm in required_metrics):
            return False
        
        return True
    
    def transform_runs_badge(df):
        for i in range(len(required_metrics)):
            df[required_metrics_name[i]] = \
                df.apply(lambda r: r['summary'][required_metrics[i]], axis=1)

        df['classes'] = df.apply(lambda r: r['config']['num_classes'], axis=1)
        df['method'] = df.apply(lambda r: f"{r['config']['mode']} {r['config']['criterium']}", axis=1)

        df = df.drop(['name', 'summary', 'config'], axis=1)

        return df

    df_badge = get_runs('badge-sampling')
    filtered_rows = df_badge.apply(filter_run_badge, axis=1)
    df_badge = df_badge[filtered_rows]
    df_badge = transform_runs_badge(df_badge)

    # joining
    df = pd.concat([df_random, df_badge], ignore_index=True)

    # computing statistics
    df_mean = df.groupby(['method']).mean()
    df_mean = df_mean.sort_values(by='f1 macro', ascending=False)
    df_mean_style = df_mean.style.highlight_max(color='lightgray').format(precision=2)

    df_std = df.groupby(['method']).std()
    df_std = df_std.loc[df_mean.index] 
    df_std_style = df_std.style.highlight_min(color='lightgray').format(precision=2)

    # combining statistics
    def round_and_convert(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns.to_list():
            df[c] = df[c].map('{:.2f}'.format)

        return df

    df_mean_std = round_and_convert(df_mean) + u'\u00B1' + round_and_convert(df_std)

    # exporting
    if path_mean is not None:
        dfi.export(df_mean_style, path_mean)

    if path_std is not None:
        dfi.export(df_std_style, path_std)

    if path_mean_std is not None:
        dfi.export(df_mean_std, path_mean_std)

def umap_features(config_path: str):
    config = load_yaml_as_obj(config_path)

    # loading dataset
    dataset_class = get_dataset_class_by_name(config.dataset)
    dataset = dataset_class('train', load_images=False, features_path=config.features_path)
    if config.standard_scale_features:
        dataset.standard_scale_features()

    # getting features to numpy
    xs, ys, zs = [], [], []
    for i in range(len(dataset.features)):
        xs.append(dataset.features[dataset.images_data['names'][i]])
        ys.append(dataset.images_data['labels'][i])
        zs.append(dataset.images_data['labels_text'][i])

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # creating the folder to save things
    os.mkdir(config.output_dir)

    def run_umap(n_neighbors, min_dist):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        mapper = reducer.fit(xs)

        image_name_prefix = f'n_neighbors={n_neighbors},min_dist={min_dist}'

        umap.plot.points(mapper, cmap='Greys')
        plt.savefig(os.path.join(config.output_dir, f'{image_name_prefix}_grey.png'), bbox_inches='tight')

        umap.plot.points(mapper, labels=zs, background='white')
        plt.savefig(os.path.join(config.output_dir, f'{image_name_prefix}_labels.png'), bbox_inches='tight')

    # creating umap images
    n_neighbors_options = [2, 5, 10, 20, 50, 100, 200, 1000]
    min_dist_options = [0.0, 0.01, 0.1, 0.25, 0.5, 0.8, 0.99]

    for n_neighbors in n_neighbors_options:
        for min_dist in min_dist_options:
            run_umap(n_neighbors, min_dist)


if __name__ == '__main__':
    task_name = sys.argv[1]
    config_path = sys.argv[2] 
    if task_name == 'umap_features':
        umap_features(config_path)
    else:
        raise ValueError(f'unknown task name: {task_name}')
    
