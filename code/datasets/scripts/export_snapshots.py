import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from simple_parsing import field

from datasets.loading import load_igraph, write_edgelist
from datasets.processing import drop_infna
from datasets.schema import DatasetSchema
from shared.cli import parse_args
from shared.constants import DatasetPath
from shared.logger import get_logger


@dataclass
class Args:
    dataset: str = field(positional=True, help="Dataset name")
    k: int = field(default=5, type=int, help="Number of snapshots to create")
    prefix: Optional[str] = field(default=None, help="Prefix for snapshot files")
    output: Optional[str] = None


args: Args = parse_args(Args)[0]
if not args.prefix:
    args.prefix = f'split_{args.k}'

LOG = get_logger(os.path.basename(__file__))
DATASET = DatasetPath(args.dataset)
schema: DatasetSchema = DatasetSchema.load_schema(args.dataset)

output_dir = DATASET.export('snapshots' if not args.output else args.output).joinpath(args.prefix)
output_dir.mkdir(parents=True, exist_ok=True)

filename = '{}_snapshot.edgelist'

# Load the graph
G = load_igraph(schema, unix_timestamp=True)
G.vs['gid'] = range(G.vcount())
G.es['gid'] = range(G.ecount())

# Set time ranges for snapshots
if schema.is_node_temporal():
    LOG.info(f'Preparing timestamps for nodes')
    interaction_types = [node_schema.label for node_schema in schema.nodes if node_schema.interaction]
    timestamps = pd.Series(G.vs['timestamp'])
    interaction_mask = pd.Series(G.vs['type']).isin(interaction_types)
    G.vs['tstart'] = timestamps.fillna(np.NINF).values
    tend = timestamps.fillna(np.PINF)
    tend[~interaction_mask] = np.PINF
    G.vs['tend'] = tend.values

if schema.is_edge_temporal():
    LOG.info('Preparing timestamps for edges')
    interaction_types = [edge_schema.type for edge_schema in schema.edges if edge_schema.interaction]
    timestamps = pd.Series(G.es['timestamp'])
    interaction_mask = pd.Series(G.es['type']).isin(interaction_types)
    G.es['tstart'] = timestamps.fillna(np.NINF)
    tend = timestamps.fillna(np.PINF)
    tend[~interaction_mask] = np.PINF
    G.es['tend'] = tend.values

# Count timestamps
LOG.info('Counting timestamps')
N_BINS = args.k * 2
timestamp_counts = None
timestamp_counts_binned = None
if schema.is_node_temporal():
    ts = pd.Series(G.vs['tstart'])
    counts = ts.value_counts()
    timestamp_counts = counts if timestamp_counts is None else timestamp_counts.add(counts, fill_value=0)
    binned_counts = drop_infna(ts).value_counts(bins=N_BINS)
    timestamp_counts_binned = binned_counts if timestamp_counts_binned is None else timestamp_counts_binned.add(
        binned_counts, fill_value=0)

if schema.is_edge_temporal():
    ts = pd.Series(G.es['tstart'])
    counts = ts.value_counts()
    timestamp_counts = counts if timestamp_counts is None else timestamp_counts.add(counts, fill_value=0)
    binned_counts = drop_infna(ts).value_counts(bins=N_BINS)
    timestamp_counts_binned = binned_counts if timestamp_counts_binned is None else timestamp_counts_binned.add(
        binned_counts, fill_value=0)

if np.NINF in timestamp_counts.index:
    timestamp_counts.drop(np.NINF, inplace=True)
if np.PINF in timestamp_counts.index:
    timestamp_counts.drop(np.PINF, inplace=True)

timestamp_counts_binned.plot.bar()
plt.show()

# Create snapshot ranges
LOG.info('Creating snapshot ranges')
timestamp_min = timestamp_counts.index.min()
timestamp_max = timestamp_counts.index.max()
snapshot_ranges = list(zip(
    np.linspace(timestamp_min, timestamp_max, args.k + 1)[:-1],
    np.linspace(timestamp_min, timestamp_max, args.k + 1)[1:]
))
print(pd.DataFrame(snapshot_ranges, columns=['tstart', 'tend']))

# Create snapshots
for i, (snapshot_start, snapshot_end) in enumerate(snapshot_ranges):
    LOG.info(f'Creating snapshot {i} - range {snapshot_start} - {snapshot_end}')
    nodes = G.vs.select(tstart_le=snapshot_end, tend_ge=snapshot_start) if schema.is_node_temporal() else G.vs
    edges = G.es.select(tstart_le=snapshot_end, tend_ge=snapshot_start) if schema.is_edge_temporal() else G.es
    G_snapshot = G.induced_subgraph(nodes).subgraph_edges(edges)
    gids = G_snapshot.vs['gid']
    edges = {
        (gids[s], gids[t])
        for (s, t) in G_snapshot.get_edgelist()
    }

    LOG.info(f'Writing snapshot {i} to {output_dir.joinpath(filename.format(i))}')
    with output_dir.joinpath(filename.format(i)).open('w') as f:
        write_edgelist(edges, f)
