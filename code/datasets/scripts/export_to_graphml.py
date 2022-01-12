import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import neo4j
from simple_parsing import ArgumentParser

from datasets.schema import DatasetSchema
from shared.config import ConnectionConfig
from shared.constants import DatasetPath
from shared.logger import get_logger


@dataclass
class Args:
    dataset: str
    edges: Optional[List[str]] = None
    split: bool = False
    output: Optional[str] = None


parser = ArgumentParser()
parser.add_arguments(Args, dest="args")
args: Args = parser.parse_args().args
connections = ConnectionConfig.load_config()

LOG = get_logger(os.path.basename(__file__))
DATASET = DatasetPath(args.dataset)
schema: DatasetSchema = DatasetSchema.load_schema(args.dataset)

export_options = {
    'format': 'gephi',
    'caption': [property.name for property in schema.all_properties() if property.label],
    'useTypes': True,
    'separateFiles': args.split,
}

output_file = DATASET.export_str('graphml', 'default.graphml' if not args.output else args.output)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if os.path.exists(output_file):
    proceed = input(f'Do you want to overwrite {output_file}? [y/N] ')
    if proceed.lower() != 'y':
        sys.exit(0)

driver = connections.neo4j.open()
with driver.session(database=schema.database) as session:
    session: neo4j.Session = session
    LOG.info(f'Exporting {schema.database} to {output_file}')
    if not args.edges:
        result = session.run(
            query='CALL apoc.export.graphml.all($output_file, $export_options)',
            parameters={'output_file': output_file, 'export_options': export_options}
        )
    else:
        edge_labels = [s.type for s in schema.edges if s.type in args.edges]
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
        LOG.info(f'Successfully exported {schema.database} to {output_file}')