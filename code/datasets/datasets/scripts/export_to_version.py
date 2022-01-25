import os
import shutil
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from simple_parsing import field

from datasets.graph_processing import graph_split_snapshot_ranges, graph_add_timeranges, graph_remove_loose_nodes
from shared.cli import parse_args
from shared.graph import DataGraph, CommunityAssignment
from shared.logger import get_logger
from shared.schema import DatasetSchema, GraphSchema, DatasetVersionType, TAG_GROUND_TRUTH

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    dataset: str = field(positional=True, help="Dataset name")
    version: Optional[str] = field(default=None, help="Dataset version")
    force: Optional[bool] = field(default=False, help="Force overwrite")
    save_graphml: Optional[bool] = field(default=True, help="Save graphml")


DEFAULT_CC_FILTER = 4


def run(args: Args):
    DATASET = DatasetSchema.load_schema(args.dataset)
    VERSION = DATASET.get_version(args.version)
    schema = GraphSchema.from_dataset(DATASET)

    if DATASET.is_synthetic():
        LOG.warning(f'Dataset {DATASET.name} is synthetic, skipping')
        return

    if VERSION.get_path().exists() and not os.getenv('FORCE'):
        LOG.info(f"Dataset version {args.dataset}:{args.version} already exists. Use --force to overwrite.")
        return

    # Delete existing data
    if VERSION.get_path().exists():
        shutil.rmtree(VERSION.get_path())

    # Load the graph
    LOG.info(f"Loading datagraph")
    G = DataGraph.from_schema(
        schema,
        unix_timestamp=True,
        prefix_id=True if 'DBLP-V' in DATASET.name else False
    )

    if VERSION.get_param('subsample', None) is not None:
        LOG.info(f"Subsampling graph to {VERSION.get_param('subsample') * 100}% nodes")
        G = G.subsample_nodes(VERSION.get_param('subsample'))

    if VERSION.get_param('cc_filter', DEFAULT_CC_FILTER) is not None:
        LOG.info(f"Filtering graph connected components. Min size: {VERSION.get_param('cc_filter', DEFAULT_CC_FILTER)}")
        G.filter_connected_components(VERSION.get_param('cc_filter', DEFAULT_CC_FILTER))

    LOG.info('Removing loose nodes')
    graph_remove_loose_nodes(G)

    TRAIN_PART = VERSION.train
    TRAIN_PART.get_path().mkdir(parents=True, exist_ok=True)

    # TODO: Add a way to filter away small components
    # component_size = pd.Series(map(len, G.components('weak')))

    # Lock down node mapping. After this, we can't change the node graph anymore
    LOG.info('Adding GIDs')
    G.add_gids()

    LOG.info(f'Saving node mapping to {TRAIN_PART.nodemapping}')
    G.save_nodemapping(str(TRAIN_PART.nodemapping))
    nodemapping = G.to_nodemapping()

    if TAG_GROUND_TRUTH in DATASET.tags:
        LOG.info(f'Loading ground-truth')
        comms = CommunityAssignment.load_comlist(str(DATASET.processed('ground_truth.ncomlist')))
        comms = comms.remap_nodes(nodemapping).renumber_communities()
        comms.named = False

    if VERSION.type == DatasetVersionType.EDGELIST_SNAPSHOTS:
        LOG.info(f'Setting time ranges for snapshots')
        graph_add_timeranges(schema, G)

        LOG.info(f'Generating snapshot ranges')
        snapshot_ranges = graph_split_snapshot_ranges(
            schema, G,
            VERSION.get_param('snapshot_count', 5),
            VERSION.get_param('snapshot_coverage', 0.95),
            plot=str(VERSION.train.get_path().joinpath('snapshot_ranges.png'))
        )
        print(pd.DataFrame(snapshot_ranges, columns=['tstart', 'tend']))

        # Create snapshots
        for i, G_snapshot in G.to_snapshots(snapshot_ranges):
            output_file = TRAIN_PART.snapshot_edgelist(i)
            LOG.info(f'Writing snapshot {i} to {output_file}')
            G_snapshot.save_edgelist(str(output_file))

            if TAG_GROUND_TRUTH in DATASET.tags:
                LOG.info(f'Saving ground-truth to {output_file.with_suffix(".comlist")}')
                comms_snapshot = comms.filter_nodes(G_snapshot.vs['gid'])
                comms_snapshot.save_comlist(str(TRAIN_PART.snapshot_ground_truth(i)))

            if args.save_graphml:
                LOG.info(f'Saving graphml to {output_file.with_suffix(".graphml")}')
                G_snapshot.write_graphml(str(output_file.with_suffix(".graphml")))

    elif VERSION.type == DatasetVersionType.EDGELIST_STATIC:
        LOG.info(f'Saving static graph to {TRAIN_PART.static_edgelist}')
        G.save_edgelist(str(TRAIN_PART.static_edgelist))

        if TAG_GROUND_TRUTH in DATASET.tags:
            LOG.info(f'Saving ground-truth to {TRAIN_PART.static_ground_truth}')
            comms.save_comlist(str(TRAIN_PART.static_ground_truth))

        if args.save_graphml:
            LOG.info(f'Saving graphml to {TRAIN_PART.static_edgelist.with_suffix(".graphml")}')
            G.write_graphml(str(TRAIN_PART.static_edgelist.with_suffix(".graphml")))


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    if args.version is None:
        DATASET = DatasetSchema.load_schema(args.dataset)
        for version in DATASET.versions.keys():
            LOG.info(f'Exporting version {args.dataset}:{version}')
            args.version = version
            run(args)
    else:
        run(args)
