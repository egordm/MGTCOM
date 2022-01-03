import os.path
import subprocess
import sys
from dataclasses import dataclass

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from simple_parsing import ArgumentParser

from datasets.schema import DatasetSchema
from shared.config import ConnectionConfig
from shared.constants import DatasetPath, DATASETS_PATH
from shared.logger import get_logger


@dataclass
class Args:
    dataset: str


parser = ArgumentParser()
parser.add_arguments(Args, dest="args")
args: Args = parser.parse_args().args
connections = ConnectionConfig.load_config()

LOG = get_logger(os.path.basename(__file__))
DATASET = DatasetPath(args.dataset)
schema = DatasetSchema.load_schema(args.dataset)

driver = connections.neo4j.open()

# Check if the database already exists
with driver.session(database='system') as session:
    # Check if database already exists
    result = session.run(f'SHOW DATABASE `{schema.database}`')
    if result.peek():
        LOG.info(f'Database {schema.database} already exists')
        proceed = input(f'Do you want to overwrite {schema.database}? [y/N] ')
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


def properties_to_neo4j(properties):
    select = []

    for property in properties:
        if property.ignore:
            continue

        if property.is_id():
            select.append(F.col(property.name).alias(f'{property.name}:ID'))
        elif property.is_src():
            select.append(F.col(property.name).alias(f':START_ID'))
        elif property.is_dst():
            select.append(F.col(property.name).alias(f':END_ID'))
        elif property.is_array():
            select.append(F.concat_ws(';', F.col(property.name)).alias(f'{property.name}:{property.type}'))
        else:
            select.append(F.col(property.name).alias(f'{property.name}:{property.type}'))
    return select


def export_csv(input: str, output: str, mapping):
    spark.read.parquet(input) \
        .select(mapping) \
        .repartition(1) \
        .write.csv(output, mode='overwrite', header=True, quoteAll=True, compression="gzip")


nodes = []
for node_schema in schema.nodes:
    path = DATASETS_PATH.joinpath(node_schema.path)
    output_path = DATASET.export('neo4j', f'{node_schema.label}.csv.gz')
    LOG.info(f'Exporting NODES {node_schema.label} to {output_path}')

    mapping = properties_to_neo4j(node_schema.properties)
    mapping.append(F.lit(node_schema.label).alias(':LABEL'))
    export_csv(str(path), str(output_path), mapping)

    nodes.extend(map(str, output_path.glob('*.csv.gz')))

relationships = []
for edge_schema in schema.edges:
    path = DATASETS_PATH.joinpath(edge_schema.path)
    output_path = DATASET.export('neo4j', f'{edge_schema.type}.csv.gz')
    LOG.info(f'Exporting EDGES {edge_schema.type} to {output_path}')

    mapping = properties_to_neo4j(edge_schema.properties)
    mapping.append(F.lit(edge_schema.type).alias(':TYPE'))
    export_csv(str(path), str(output_path), mapping)
    relationships.extend(map(str, output_path.glob('*.csv.gz')))

with driver.session(database='system') as session:
    LOG.info('Stopping and deleting existing database')
    try:
        session.run(f'STOP DATABASE `{schema.database}`')
        session.run(f'DROP DATABASE `{schema.database}`')
    except Exception as e:
        LOG.warning(f'Failed to stop and delete existing database: {e}')

    LOG.info('Creating new database from data')
    node_args = [f'--nodes="{s}"' for s in nodes]
    relationship_args = [f'--relationships="{s}"' for s in relationships]
    p = subprocess.Popen(
        'bin/neo4j-admin import --database={database} --legacy-style-quoting=true --multiline-fields=true '
        '--verbose --trim-strings=false --ignore-empty-strings=true --normalize-types=true '
        '--skip-bad-relationships=true --skip-duplicate-nodes=true {nodes} {relationships}'.format(
            database=schema.database,
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
        session.run(f'CREATE DATABASE `{schema.database}`')
    except Exception as e:
        LOG.warning(f'Failed to create new database: {e}')

LOG.info(f'Connecting to database: {schema.database}')
with driver.session(database=schema.database) as session:
    LOG.info('Creating contraints')
    for node_schema in schema.nodes:
        try:
            session.run(
                f'CREATE CONSTRAINT {node_schema.label}_unique_id ON (n:{node_schema.label}) ASSERT n.id IS UNIQUE')
        except Exception as e:
            LOG.error(f'Failed to create constraint: {e}')
