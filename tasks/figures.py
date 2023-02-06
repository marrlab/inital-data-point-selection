
import wandb
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
from utils.utils import load_dataframes
from scipy.stats import entropy
from utils.types import Result
from utils.utils import result_to_dataframe
from collections import defaultdict


def subsetting_methods_performance_preprocessing(result: Result) -> pd.DataFrame:
    df = result_to_dataframe(result)
    df = df.groupby(['method']).mean()
    df = df.div(df.sum(axis=1), axis=0)

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


def classification_metrics():
    api = wandb.Api()

    runs = api.runs('mireczech/random-baseline')
    hist_list = [] 
    for run in runs: 
        # if not 'val/loss' in run.summary:
        #     continue

        # name = run.config['model']['_target_'].split('.')[-1]
        hist = run.history(keys=['epoch', 'val_balanced_accuracy'])
        hist_list.append(hist)

    df = pd.concat(hist_list, ignore_index=True)
    # df = df.query("`val/loss` != 'NaN'")

    # sns.lineplot(x="epoch", y="val/loss", hue="name", data=df)
    # plt.show()

    return df