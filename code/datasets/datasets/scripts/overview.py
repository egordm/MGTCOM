import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from datasets.files import get_dataset_names
from shared.cli import parse_args
from shared.constants import REPORTS_PATH
from shared.graph import pd_from_graph_schema
from shared.logger import get_logger
from shared.schema import DatasetSchema, TAG_DYNAMIC, TAG_GROUND_TRUTH, TAG_OVERLAPPING, GraphSchema, DatasetVersionType

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    with_extra: bool = True
    with_versions: bool = False


def run(args: Args):
    datasets = [
        DatasetSchema.load_schema(name)
        for name in get_dataset_names()
    ]

    df = pd.DataFrame({
        'name': [d.name for d in datasets],
    }).set_index('name')

    df['type'] = ['dynamic' if TAG_DYNAMIC in d.tags else 'static' for d in datasets]
    df['ground_truth'] = [
        ('y:overlapping' if TAG_OVERLAPPING in d.tags else 'y:partition') if TAG_GROUND_TRUTH in d.tags else 'no'
        for d in datasets
    ]

    extra = []
    for d in datasets:
        info = {}
        LOG.info('Checking graph for dataset %s', d.name)
        schema = GraphSchema.from_dataset(d)
        nodes_df, edges_df = pd_from_graph_schema(
            schema,
            unix_timestamp=False,
        )
        info['dataset'] = d.name
        info['node_count'] = len(nodes_df)
        info['node_count'] = len(edges_df)
        info['interaction'] = 'y' if schema.is_node_interaction() else 'n'
        info['type_counts'] = f'{len(schema.nodes)}/{len(schema.edges)}'
        extra.append(info)

    df_extra = pd.DataFrame(extra).set_index('dataset')
    df = pd.merge(df, df_extra, left_index=True, right_index=True)

    if args.with_versions:
        df_versions = pd.concat([
            load_version_info(d)
            for d in datasets
        ]).add_prefix('version_')

        df = pd.merge(df, df_versions, left_index=True, right_index=True)

    df.fillna('-', inplace=True)
    df.index.name = 'dataset'

    REPORTS_PATH.mkdir(exist_ok=True)
    filename = 'dataset_versions' if args.with_versions else 'datasets'
    df.to_csv(REPORTS_PATH.joinpath(f'{filename}.csv'), index=True)
    df.to_markdown(REPORTS_PATH.joinpath(f'{filename}.md'), index=True)
    print(df.to_markdown())


def load_version_info(dataset: DatasetSchema):
    data = []
    for name, version in dataset.versions.items():
        LOG.info('Checking version %s for dataset %s', name, dataset.name)
        item = {
            'dataset': dataset.name,
            'name': name,
            'format': version.type.pretty(),
        }

        if version.type == DatasetVersionType.EDGELIST_SNAPSHOTS:
            n_snapshots = version.get_param('snapshot_count', -1)
            item['snapshots'] = n_snapshots
            node_count, edge_count = [], []
            for i in range(n_snapshots):
                graph_info = yaml.safe_load(
                    version.train.snapshot_ground_truth(i).with_suffix('.info.yaml').read_text())
                node_count.append(graph_info['nodes'])
                edge_count.append(graph_info['edges'])

            item['node_count'] = f'{np.mean(node_count):.0f} ± {np.std(node_count):.0f}'
            item['edge_count'] = f'{np.mean(edge_count):.0f} ± {np.std(edge_count):.0f}'

        elif version.type == DatasetVersionType.EDGELIST_STATIC:
            graph_info = yaml.safe_load(version.train.static_edgelist.with_suffix('.info.yaml').read_text())
            item['node_count'] = graph_info['nodes']
            item['edge_count'] = graph_info['edges']

        data.append(item)

    return pd.DataFrame(data).set_index('dataset')


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    run(args)
