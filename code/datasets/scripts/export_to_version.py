import os
from dataclasses import dataclass

import pandas as pd
from simple_parsing import field

from datasets.graph_processing import graph_split_snapshot_ranges, graph_add_timeranges, graph_split_into_snapshots
from shared.cli import parse_args
from shared.graph import DataGraph, igraph_to_edgelist, write_edgelist, CommunityAssignment
from shared.logger import get_logger
from shared.schema import DatasetSchema, GraphSchema, DatasetVersionType


@dataclass
class Args:
    dataset: str = field(positional=True, help="Dataset name")
    version: str = field(positional=True, help="Dataset version")


args: Args = parse_args(Args)[0]

LOG = get_logger(os.path.basename(__file__))
DATASET = DatasetSchema.load_schema(args.dataset)
VERSION = DATASET.get_version(args.version)
schema = GraphSchema.from_dataset(DATASET)

# Load the graph
LOG.info(f"Loading datagraph")
G = DataGraph.from_schema(schema, unix_timestamp=True)

LOG.info('Adding GIDs')
G.add_gids()

VERSION.train_part().get_path().mkdir(parents=True, exist_ok=True)

# TODO: Add a way to filter away small components
# component_size = pd.Series(map(len, G.components('weak')))

# Lock down node mapping. After this, we can't change the node graph anymore
LOG.info(f'Saved node mapping to {VERSION.train_part().nodemapping}')
G.save_nodemapping(str(VERSION.train_part().nodemapping))
nodemapping = G.to_nodemapping()

if VERSION.type == DatasetVersionType.SNAPSHOTS:
    LOG.info(f'Setting time ranges for snapshots')
    graph_add_timeranges(schema, G)

    LOG.info(f'Generating snapshot ranges')
    snapshot_ranges = graph_split_snapshot_ranges(
        schema, G, VERSION.get_param('snapshot_count', 5), VERSION.get_param('snapshot_coverage', 0.95), plot=True
    )
    print(pd.DataFrame(snapshot_ranges, columns=['tstart', 'tend']))

    # Create snapshots
    VERSION.train_part().snapshots.mkdir(exist_ok=True, parents=True)
    for i, G_snapshot in G.to_snapshots(snapshot_ranges):
        filename = f'{str(i).zfill(2)}_snapshot.edgelist'
        output_file = VERSION.train_part().snapshots.joinpath(filename)

        LOG.info(f'Writing snapshot {i} to {output_file}')
        G_snapshot.save_edgelist(str(output_file))

elif VERSION.type == DatasetVersionType.STATIC:
    LOG.info(f'Saving static graph to {VERSION.train_part().static}')
    G.save_edgelist(str(VERSION.train_part().static))

    if 'ground-truth' in DATASET.tags:
        LOG.info(f'Saving ground-truth to {VERSION.train_part().ground_truth}')
        comms = CommunityAssignment.load_comlist(str(DATASET.processed('ground_truth.ncomlist')), named=True)
        comms = comms.remap_nodes(nodemapping).renumber_communities()
        comms.named = False
        comms.save_comlist(str(VERSION.train_part().ground_truth))

