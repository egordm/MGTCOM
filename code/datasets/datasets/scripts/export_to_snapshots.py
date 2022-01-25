import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from simple_parsing import field

from datasets.graph_processing import graph_add_timeranges, graph_split_into_snapshots
from datasets.visualization import plot_value_distribution_histogram
from shared.cli import parse_args
from shared.graph import DataGraph, igraph_to_edgelist, write_edgelist
from shared.logger import get_logger
from shared.pandas import drop_infna
from shared.schema import DatasetSchema, GraphSchema


@dataclass
class Args:
    dataset: str = field(positional=True, help="Dataset name")
    k: int = field(default=5, type=int, help="Number of snapshots to create")
    prefix: Optional[str] = field(default=None, help="Prefix for snapshot files")
    output: Optional[str] = None


def run(args: Args):
    if not args.prefix:
        args.prefix = f'split_{args.k}'

    LOG = get_logger(os.path.basename(__file__))
    DATASET = DatasetSchema(args.dataset)
    schema = GraphSchema.from_dataset(args.dataset)

    output_dir = DATASET.export('snapshots' if not args.output else args.output).joinpath(args.prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = '{}_snapshot.edgelist'

    # Load the graph
    G = DataGraph.from_schema(schema, unix_timestamp=True)
    G.vs['gid'] = range(G.vcount())
    G.es['gid'] = range(G.ecount())

    LOG.info(f'Setting time ranges for snapshots')
    graph_add_timeranges(schema, G)

    LOG.info('Counting timestamps')
    timestamps_df = None
    if schema.is_node_temporal():
        timestamps_df = pd.Series(G.vs['tstart'])
    if schema.is_edge_temporal():
        timestamps_df = pd.Series(G.es['tstart']) if timestamps_df is None \
            else pd.concat([timestamps_df, pd.Series(G.es['tstart'])])

    timestamps_df = drop_infna(timestamps_df)

    # Count timestamps
    plot_value_distribution_histogram(
        timestamps_df,
        bins=args.k * 2,
        title=f'{args.dataset} - Timestamps',
        xlabel='Time',
        ylabel='Node+Edge Count',
    )
    plt.show()

    # Create snapshot ranges
    LOG.info('Creating snapshot ranges')
    timestamp_min = timestamps_df.min()
    timestamp_max = timestamps_df.max()
    snapshot_ranges = list(zip(
        np.linspace(timestamp_min, timestamp_max, args.k + 1)[:-1],
        np.linspace(timestamp_min, timestamp_max, args.k + 1)[1:]
    ))
    print(pd.DataFrame(snapshot_ranges, columns=['tstart', 'tend']))

    # Create snapshots
    for i, G_snapshot in graph_split_into_snapshots(schema, G, snapshot_ranges):
        output_file = output_dir.joinpath(filename.format(str(i).zfill(2)))
        LOG.info(f'Writing snapshot {i} to {output_file}')
        edges = igraph_to_edgelist(G_snapshot)
        write_edgelist(edges, str(output_file))


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    run(args)
