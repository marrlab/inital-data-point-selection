
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi
from scipy.stats import entropy
from utils.types import Result
from utils.utils import result_to_dataframe


def subsetting_methods_performance_preprocessing(result: Result) -> pd.DataFrame:
    df = result_to_dataframe(result)
    df = df.groupby(['method']).mean()
    df = df.div(df.sum(axis=1), axis=0)

    return df


def subsetting_methods_performance_heatmap(result: Result, path: str = None):
    df = subsetting_methods_performance_preprocessing(result)
    df *= 100

    _, ax = plt.subplots(figsize=(15,5))

    sns.heatmap(df, annot=True, fmt='.1f', linewidths=.5, cmap='gray_r', ax=ax)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def subsetting_methods_performance_entropy(result: Result, path: str):
    df = subsetting_methods_performance_preprocessing(result)

    df['entropy'] = df.apply(lambda x: entropy(x), axis=1)
    df = df[['entropy']]
    dfi.export(df, path)


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