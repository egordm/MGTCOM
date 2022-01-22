import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_value_distribution_histogram(
        series: pd.Series,
        bins: int = 20,
        title: str = '',
        xlabel: str = '',
        ylabel: str = 'Count',
):
    ax = series.dropna().hist(
        bins=max(1, min(bins, len(series.dropna().value_counts()))),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return ax


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

    data = series[series != 0]
    ax = sns.histplot(
        data=data,
        bins=max(1, min(bins[1], len(data.value_counts()))),
        cumulative=True,
        ax=axes[1],
        log_scale=(True, False),
        stat='density' if normalize else 'count',
    )
    ax.set_title(f'Cumulative Log {title_short or title}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, axes


def show_top_k_nodes(
        series: pd.Series,
        labels: dict = None,
        title: str = '',
        k: int = 10,
        show=True,
):
    top_k = series.sort_values(ascending=False).head(k).index

    if show:
        print('==============================')
        print(f'Top {k} {title}')
    df_tmp = pd.DataFrame([
        {
            'value': series[i],
            'label': labels[i],
        }
        for i in top_k
    ])
    if show:
        display(df_tmp.head(k))

    if not show:
        return df_tmp


def show_top_k_stacked_nodes(
        data: pd.DataFrame,
        labels: dict = None,
        title: str = '',
        k: int = 10,
        show=True,
        item_title='Item',
):
    dfs = []
    for col in data.columns:
        col_title = col.replace('_', ' ').title()
        df = show_top_k_nodes(
            data[col],
            labels,
            title=col_title,
            show=False
        ).rename(columns={'value': col_title, 'label': f'{col_title} {item_title}'})
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    if show:
        print('==============================')
        print(f'Top {k} {title}')
        display(df.head(k))
    else:
        return df
