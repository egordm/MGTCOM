import os.path
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from simple_parsing import field

from shared.cli import parse_args
from shared.config import ConnectionConfig
from shared.constants import DATASETS_PATH
from shared.logger import get_logger
from shared.schema import GraphSchema, DatasetSchema, GraphProperty


@dataclass
class Args:
    dataset: str = field(positional=True)


def run(args: Args):
    connections = ConnectionConfig.load_config()

    LOG = get_logger(os.path.basename(__file__))
    DATASET = DatasetSchema.load_schema(args.dataset)
    schema = GraphSchema.from_dataset(DATASET)

    driver = connections.neo4j.open()

    # Check if the database already exists
    with driver.session(database='system') as session:
        # Check if database already exists
        result = session.run(f'SHOW DATABASE `{DATASET.database}`')
        if result.peek() and not os.getenv('FORCE', True):
            LOG.info(f'Database {DATASET.database} already exists')
            proceed = input(f'Do you want to overwrite {DATASET.database}? [y/N] ')
            if proceed.lower() != 'y':
                sys.exit(0)

    spark = (SparkSession.builder
             .appName(f'{DATASET}_preprocess')
             .config('spark.sql.legacy.timeParserPolicy', 'LEGACY')
             .config("spark.executor.memory", "8g")
             .config("spark.driver.memory", "8g")
             .config("spark.memory.offHeap.enabled", True)
             .config("spark.memory.offHeap.size", "16g")
             .getOrCreate())

    def properties_to_neo4j(properties: Iterable[GraphProperty]):
        select = []

        for property in properties:
            if property.name == 'id':
                select.append(F.col(property.name).alias(f'{property.name}:ID'))
            elif property.name == 'src':
                select.append(F.col(property.name).alias(f':START_ID'))
            elif property.name == 'dst':
                select.append(F.col(property.name).alias(f':END_ID'))
            elif property.dtype.array:
                select.append(F.concat_ws(';', F.col(property.name)).alias(f'{property.name}:{property.dtype}'))
            else:
                select.append(F.col(property.name).alias(f'{property.name}:{property.dtype}'))
        return select

    def export_csv(input: str, output: str, mapping):
        spark.read.parquet(input) \
            .select(mapping) \
            .repartition(1) \
            .write.csv(output, mode='overwrite', header=True, quoteAll=True, compression="gzip")

    nodes = []
    for node_schema in schema.nodes.values():
        path = DATASETS_PATH.joinpath(node_schema.get_path())
        output_path = DATASET.export('neo4j', f'{node_schema.get_type()}.csv.gz')
        LOG.info(f'Exporting NODES {node_schema.get_type()} to {output_path}')

        mapping = properties_to_neo4j(node_schema.properties.values())
        mapping.append(F.lit(node_schema.label).alias(':LABEL'))
        export_csv(str(path), str(output_path), mapping)

        nodes.extend(map(str, output_path.glob('*.csv.gz')))

    relationships = []
    for edge_schema in schema.edges.values():
        path = DATASETS_PATH.joinpath(edge_schema.get_path())
        output_path = DATASET.export('neo4j', f'{edge_schema.get_type()}.csv.gz')
        LOG.info(f'Exporting EDGES {edge_schema.get_type()} to {output_path}')

        mapping = properties_to_neo4j(edge_schema.properties.values())
        mapping.append(F.lit(edge_schema.get_type()).alias(':TYPE'))
        export_csv(str(path), str(output_path), mapping)
        relationships.extend(map(str, output_path.glob('*.csv.gz')))

    with driver.session(database='system') as session:
        LOG.info('Stopping and deleting existing database')
        try:
            session.run(f'STOP DATABASE `{DATASET.database}`')
            session.run(f'DROP DATABASE `{DATASET.database}`')
        except Exception as e:
            LOG.warning(f'Failed to stop and delete existing database: {e}')

        LOG.info('Creating new database from data')
        node_args = [f'--nodes="{s}"' for s in nodes]
        relationship_args = [f'--relationships="{s}"' for s in relationships]
        p = subprocess.Popen(
            'bin/neo4j-admin import --database={database} --legacy-style-quoting=true --multiline-fields=true '
            '--verbose --trim-strings=false --ignore-empty-strings=true --normalize-types=true '
            '--skip-bad-relationships=true --skip-duplicate-nodes=true {nodes} {relationships}'.format(
                database=DATASET.database,
                nodes=' '.join(node_args),
                relationships=' '.join(relationship_args),
            ),
            shell=True,
            stdout=sys.stdout, stderr=sys.stderr,
            cwd=str(connections.neo4j.neo4j_home),
            env={
                'JAVA_HOME': connections.neo4j.java_home,
                'NEO4J_HOME': connections.neo4j.neo4j_home,
            }
        )
        p.wait()
        if p.returncode != 0:
            LOG.error(f'Failed to import data to neo4j: {p.returncode}')
            sys.exit(p.returncode)

        LOG.info('Creating database and indexes')
        try:
            session.run(f'CREATE DATABASE `{DATASET.database}`')
        except Exception as e:
            LOG.warning(f'Failed to create new database: {e}')

    LOG.info(f'Connecting to database: {DATASET.database}')
    with driver.session(database=DATASET.database) as session:
        LOG.info('Creating contraints')
        for node_schema in schema.nodes.values():
            try:
                session.run(
                    f'CREATE CONSTRAINT {node_schema.get_type()}_unique_id ON (n:{node_schema.get_type()}) ASSERT n.id IS UNIQUE')
            except Exception as e:
                LOG.error(f'Failed to create constraint: {e}')


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    run(args)
