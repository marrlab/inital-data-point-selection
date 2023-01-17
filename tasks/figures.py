
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def subsetting_methods_performance(results: pd.DataFrame, path: str = None):
    sns.heatmap(results.groupby(['method']).mean(), annot=True, linewidths=.5, cmap='gray_r')

    if path is not None:
        plt.savefig(path, bbox_inches='tight')

    plt.show()
