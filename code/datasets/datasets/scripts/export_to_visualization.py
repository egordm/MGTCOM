import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import igraph as ig
import numpy as np
from simple_parsing import field

from datasets.graph_processing import graph_join_attribute
from shared.cli import parse_args
from shared.graph import CommunityAssignment, DataGraph
from shared.logger import get_logger
from shared.schema import DatasetSchema, DatasetVersionType, GraphSchema

LOG = get_logger(os.path.basename(__file__))


@dataclass
class Args:
    dataset: str = field(positional=True, help="Dataset name")
    version: Optional[str] = field(alias='v', default=None, help="Dataset version")
    run_paths: List[str] = field(alias='r', default_factory=list, help="run name")


DEFAULT_CC_FILTER = 4


def run(args: Args):
    DATASET = DatasetSchema.load_schema(args.dataset)
    VERSION = DATASET.get_version(args.version)

    export_dir = DATASET.export('versions', args.version)
    export_dir.mkdir(parents=True, exist_ok=True)

    def process_graph(graph_file: Path):
        LOG.info(f"Processing {graph_file}")
        if graph_file.suffix == '.graphml':
            graph = ig.Graph.Read_GraphML(str(graph_file.with_suffix(".graphml")))
        elif graph_file.name == 'schema.yaml':
            schema = GraphSchema.load_schema(graph_file)
            graph = DataGraph.from_schema(
                schema,
                explicit_timestamp=True,
                unix_timestamp=True,
                include_properties=lambda xs: [x for x in xs if not x.startswith('feat_')]
            )
            graph.write_graphml(str('aa'))
            graph.vs['gid'] = graph.vs['name']
            # if schema.is_node_temporal() or schema.is_edge_temporal():
            #     graph.add_timeranges()
        else:
            raise ValueError(f"Unknown file type: {graph_file}")

        # TODO: make this configurable
        if 'weight' in graph.es.attributes():
            graph.es['weight'] = 1

        if 'tstart' in graph.vs.attributes():
            graph.vs['tstart'] = [
                t if isinstance(t, (int, float)) and np.isfinite(t) else 0
                for t in graph.vs['tstart']
            ]
            graph.vs['tend'] = [
                t if isinstance(t, (int, float)) and np.isfinite(t) else 2147483646
                for t in graph.vs['tend']
            ]

        if 'tstart' in graph.es.attributes():
            graph.es['tstart'] = [
                t if isinstance(t, (int, float)) and np.isfinite(t) else 0
                for t in graph.es['tstart']
            ]
            graph.es['tend'] = [
                t if isinstance(t, (int, float)) and np.isfinite(t) else 2147483646
                for t in graph.es['tend']
            ]

        if DATASET.has_groundtruth():
            LOG.info("Adding ground truth")
            ground_truth = CommunityAssignment.load_comlist(str(graph_file.with_suffix(".comlist")))
            ground_truth.with_nodes(graph.vs['gid'])
            graph_join_attribute(graph, ground_truth.data['cid'], 'ground_truth', left_attribute='gid', group=False)

        for i, path in enumerate(map(Path, args.run_paths)):
            prediction_file = path.joinpath(graph_file.with_suffix(".comlist").name)

            if not path.exists() or not prediction_file.exists():
                LOG.warning(f"Path {prediction_file} does not exist")
                continue

            LOG.info(f"Adding prediction {prediction_file} as prediction_{i}")
            prediction = CommunityAssignment.load_comlist(str(prediction_file))
            prediction.with_nodes(graph.vs['gid'])
            graph_join_attribute(graph, prediction.data['cid'], f'prediction_{i}', left_attribute='gid', group=False)

        output_file = export_dir.joinpath(graph_file.with_suffix(".graphml").name)
        LOG.info(f"Writing {output_file}")
        graph.write_graphml(str(output_file))

    if VERSION.type == DatasetVersionType.EDGELIST_SNAPSHOTS:
        for snapshot_file in VERSION.train.get_snapshot_edgelists():
            process_graph(snapshot_file.with_suffix(".graphml"))

    elif VERSION.type == DatasetVersionType.EDGELIST_STATIC:
        process_graph(VERSION.train.static_edgelist.with_suffix(".graphml"))

    elif VERSION.type == DatasetVersionType.GRAPHSCHEMA:
        process_graph(VERSION.train.graphschema)



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
