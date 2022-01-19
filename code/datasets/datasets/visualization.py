from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_explore_dual_histogram(
        series: pd.Series,
        quantile=0.75,
        title: str = '',
        title_short: str = '',
        xlabel: str = '',
        ylabel: str = 'Count',
        bins=(10, 20),
        normalize=False,
):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle(title, fontsize=20)

    data = series[series < series.quantile(quantile)]
    ax = sns.histplot(
        data=data,
        bins=max(1, min(bins[0], len(data.value_counts()))),
        ax=axes[0],
        stat='density' if normalize else 'count',
    )
    ax.set_title(f"{title_short or title} ({int(quantile * 100)}% of the data)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    data = series
    ax = sns.histplot(
        data=data,
        bins=max(1, min(bins[1], len(data.value_counts()))),
        log_scale=True,
        ax=axes[1],
        stat='density' if normalize else 'count',
    )
    ax.set_title(f'Log {title_short or title}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, axes


def show_top_k_nodes(
        series: pd.Series,
        labels: dict = None,
        title: str = '',
        k: int = 10,
):
    top_k = series.sort_values(ascending=False).head(k).index

    print('==============================')
    print(f'Top {k} {title}')
    df_tmp = pd.DataFrame([
        {
            'value': series[i],
            'label': labels[i],
        }
        for i in top_k
    ])
    display(df_tmp.head(k))
