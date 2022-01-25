import os
import sys
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional

import neo4j
from simple_parsing import field

from shared.cli import parse_args
from shared.config import ConnectionConfig
from shared.logger import get_logger
from shared.schema import DatasetSchema, GraphSchema


@dataclass
class Args:
    dataset: str = field(positional=True)
    edges: Optional[List[str]] = None
    split: bool = False
    output: Optional[str] = None


def run(args: Args):
    connections = ConnectionConfig.load_config()

    LOG = get_logger(os.path.basename(__file__))
    DATASET = DatasetSchema.load_schema(args.dataset)
    schema = GraphSchema.from_dataset(DATASET)

    if DATASET.is_synthetic():
        LOG.warning(f'Dataset {DATASET.name} is synthetic, skipping')
        return

    export_options = {
        'format': 'gephi',
        'caption': [subschema.label for subschema in chain(schema.nodes.values(), schema.edges.values()) if
                    subschema.label],
        'useTypes': True,
        'separateFiles': args.split,
    }

    output_file = DATASET.export_str('graphml', 'default.graphml' if not args.output else args.output)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file) and not os.getenv('FORCE', True):
        proceed = input(f'Do you want to overwrite {output_file}? [y/N] ')
        if proceed.lower() != 'y':
            sys.exit(0)

    driver = connections.neo4j.open()
    with driver.session(database=DATASET.database) as session:
        session: neo4j.Session = session
        LOG.info(f'Exporting {DATASET.database} to {output_file}')
        if not args.edges:
            result = session.run(
                query='CALL apoc.export.graphml.all($output_file, $export_options)',
                parameters={'output_file': output_file, 'export_options': export_options}
            )
        else:
            edge_labels = schema.get_edge_types()
            if len(edge_labels) != len(args.edges):
                LOG.error(f'Could not find some edges: {args.edges} vs {edge_labels}')
                sys.exit(1)

            result = session.run(
                query=f'''
                MATCH (n1)-[r:{'|'.join(edge_labels)}]->(n2)
                WITH [n1, n2] as nc, r as r
                UNWIND nc as n
                WITH collect(DISTINCT n) as all_nodes, collect(r) as all_relations
                CALL apoc.export.graphml.data(all_nodes, all_relations, $output_file, $export_options)
                YIELD file, nodes, relationships, properties, data
                RETURN file, nodes, relationships, properties, data;
                ''',
                parameters={'output_file': output_file, 'export_options': export_options}
            )

        if result.peek():
            LOG.info(f'Successfully exported {DATASET.database} to {output_file}')


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    run(args)
