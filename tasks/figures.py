
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
