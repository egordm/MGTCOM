import os
import sys
from dataclasses import dataclass
from typing import Optional

import neo4j
from simple_parsing import field

from datasets.schemas import DatasetSchema
from shared.cli import parse_args
from shared.config import ConnectionConfig
from shared.constants import DatasetPath
from shared.io import remove_first_line
from shared.logger import get_logger


@dataclass
class Args:
    dataset: str = field(positional=True)
    # edges: Optional[List[str]] = None
    split: bool = False
    output: Optional[str] = None


args: Args = parse_args(Args)[0]
connections = ConnectionConfig.load_config()

LOG = get_logger(os.path.basename(__file__))
DATASET = DatasetPath(args.dataset)
schema: DatasetSchema = DatasetSchema.load_schema(args.dataset)

export_options = {
    'delim': ' ',
    'quotes': False,
    'separateHeader': True,
}

output_file = DATASET.export_str('edgelist', 'default.edgelist' if not args.output else args.output)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if os.path.exists(output_file):
    proceed = input(f'Do you want to overwrite {output_file}? [y/N] ')
    if proceed.lower() != 'y':
        sys.exit(0)

driver = connections.neo4j.open()
with driver.session(database=schema.database) as session:
    session: neo4j.Session = session
    LOG.info(f'Exporting {schema.database} to {output_file}')
    result = session.run(
        query=r'''
        WITH "
            MATCH (a)-[r]->(b)
            RETURN a.id as src, b.id as dst
        " AS query
        CALL apoc.export.csv.query(query, $output_file, $export_options)
        YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
        RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
        ''',
        parameters={'output_file': output_file, 'export_options': export_options}
    )

    if result.peek():
        LOG.info(f'Successfully exported {schema.database} to {output_file}')

    remove_first_line(output_file)
