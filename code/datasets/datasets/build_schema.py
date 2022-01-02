import os.path
import re
from typing import List, Tuple

import pyspark.sql.types as T
import stringcase
from pyspark.sql import SparkSession

from datasets.schema import DatasetSchema, Property, NodeSchema, EdgeSchema
from shared.constants import DATASETS_PATH
from shared.logger import get_logger

File = str

LOG = get_logger(__file__)


def stype_to_ntype(stype: T.DataType) -> str:
    if isinstance(stype, T.ArrayType):
        return f'{stype_to_ntype(stype.elementType)}[]'
    elif isinstance(stype, T.StringType):
        return 'string'
    elif isinstance(stype, T.IntegerType):
        return 'int'
    elif isinstance(stype, T.LongType):
        return 'long'
    elif isinstance(stype, T.FloatType):
        return 'float'
    elif isinstance(stype, T.DoubleType):
        return 'double'
    elif isinstance(stype, T.BooleanType):
        return 'boolean'
    elif isinstance(stype, T.TimestampType):
        return 'datetime'
    elif isinstance(stype, T.DateType):
        return 'date'
    else:
        raise NotImplementedError


def sstruct_to_nstruct(spark_schema: T.StructType) -> List[Property]:
    result = []
    for name, stype in zip(spark_schema.names, spark_schema.fields):
        try:
            result.append(Property(name, stype_to_ntype(stype.dataType)))
        except NotImplementedError:
            LOG.warn(f'Unsupported type: {stype.dataType}')
    return result


def df_to_node_schema(name: str, path: str, spark: SparkSession) -> NodeSchema:
    df = spark.read.parquet(path)

    return NodeSchema(
        path=os.path.relpath(path, DATASETS_PATH),
        label=stringcase.pascalcase(name),
        properties=sstruct_to_nstruct(df.schema),
    )


def df_to_edge_schema(name: str, src: str, dst: str, path: str, spark: SparkSession) -> EdgeSchema:
    df = spark.read.parquet(path)

    return EdgeSchema(
        path=os.path.relpath(path, DATASETS_PATH),
        type=stringcase.constcase(name),
        source=stringcase.pascalcase(src),
        target=stringcase.pascalcase(dst),
        properties=sstruct_to_nstruct(df.schema),
    )


def to_identifier(s: str) -> str:
    s = re.sub('[- ]', '_', s)
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)


def build_schema(
        spark: SparkSession,
        name: str,
        nodes: List[Tuple[str, File]],
        edges: List[Tuple[str, str, str, File]],
) -> DatasetSchema:
    # Sanity check naming
    node_names = [name for name, _ in nodes]
    for (_, src, dst, _) in edges:
        if src not in node_names:
            raise ValueError(f'Unknown node: {src}')
        if dst not in node_names:
            raise ValueError(f'Unknown node: {dst}')

    # Build node schema
    result = DatasetSchema(
        name=name,
        prefix=stringcase.pascalcase(to_identifier(name)),
        database=to_identifier(name).replace('_', '-'),
        description=None,
        nodes=[
            df_to_node_schema(name, path, spark)
            for (name, path) in nodes
        ],
        edges=[
            df_to_edge_schema(name, src, dst, path, spark)
            for (name, src, dst, path) in edges
        ],
    )

    # Load and merge old schema
    try:
        old_schema = DatasetSchema.load_schema(name)
        LOG.debug(f'Merging old schema for {name}')
        result = old_schema.merge(result)
    except FileNotFoundError:
        pass

    result.save_schema()
    return result
