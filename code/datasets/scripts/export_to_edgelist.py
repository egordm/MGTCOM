import os
from dataclasses import dataclass
from typing import Optional

from simple_parsing import field

from shared.cli import parse_args
from shared.graph import DataGraph, igraph_to_edgelist, write_edgelist
from shared.logger import get_logger
from shared.schema import DatasetSchema, GraphSchema


@dataclass
class Args:
    dataset: str = field(positional=True)
    output: Optional[str] = None
    prefix: Optional[str] = field(default=None, help="Prefix for snapshot files")


args: Args = parse_args(Args)[0]
if not args.prefix:
    args.prefix = f'default'

LOG = get_logger(os.path.basename(__file__))
DATASET = DatasetSchema.load_schema(args.dataset)
schema = GraphSchema.from_dataset(DATASET)

output_dir = DATASET.export('edgelist' if not args.output else args.output).joinpath(args.prefix)
output_dir.mkdir(parents=True, exist_ok=True)

# Load the graph
LOG.info('Loading graph...')
G = DataGraph.from_schema(schema, unix_timestamp=True)
G.vs['gid'] = range(G.vcount())
G.es['gid'] = range(G.ecount())

# Export to edgelist
LOG.info(f'Exporting to edgelist...')
edges = igraph_to_edgelist(G)
write_edgelist(edges, str(output_dir.joinpath('default.edgelist')))
