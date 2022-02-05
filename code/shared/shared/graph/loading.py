from typing import List, Tuple, Union, Optional, Callable

import numpy as np
import pandas as pd

from shared.schema import EntitySchema, EdgeSchema, GraphSchema


def pd_from_entity_schema(
        schema: EntitySchema,
        explicit_label: bool = True,
        explicit_timestamp: bool = True,
        include_properties: Optional[Union[List[str], Callable[[List[str]], List[str]]]] = None,
        unix_timestamp: bool = False,
        prefix_id: bool = False,
) -> pd.DataFrame:
    df = pd.read_parquet(schema.get_path(), engine='pyarrow', use_nullable_dtypes=True)

    # Move explicit properties to their own column
    if explicit_label and schema.label:
        df['label'] = df[schema.label]
    if explicit_timestamp and schema.dynamic:
        df['timestamp'] = df[schema.dynamic.timestamp]
        if unix_timestamp and str(df['timestamp'].dtype).startswith('datetime'):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.NAN)

    # Set type attribute
    df['type'] = schema.get_type()

    id_props = ['src', 'dst'] if isinstance(schema, EdgeSchema) else ['id']

    # Prefix id with schema type
    if prefix_id:
        if isinstance(schema, EdgeSchema):
            df['src'] = schema.source_type + ':' + df['src']
            df['dst'] = schema.target_type + ':' + df['dst']
        else:
            df['id'] = schema.get_type() + ':' + df['id']

    # Select only necessary columns
    if callable(include_properties):
        include_properties = include_properties(df.columns)
    props = {*id_props, 'type', 'label', 'timestamp', *(include_properties or [])} & set(df.columns)
    df.drop(columns=set(df.columns).difference(props), inplace=True)

    if 'name' in df.columns:
        df['name_'] = df['name']
        df.drop(columns=['name'], inplace=True)

    # Move id to the first column
    for key in reversed(id_props):
        ids = df.pop(key)
        df.insert(0, key, ids)

    # Add duplicated edges in case of undirected graph
    if isinstance(schema, EdgeSchema):
        if not schema.directed:
            df = pd.concat([df, df.rename(columns={'src': 'dst', 'dst': 'src'})])

    return df


def pd_from_graph_schema(
        schema: GraphSchema,
        explicit_label: bool = True,
        explicit_timestamp: bool = True,
        include_properties: Optional[Union[List[str], Callable[[List[str]], List[str]]]] = None,
        unix_timestamp: bool = False,
        prefix_id: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_dfs = [
        pd_from_entity_schema(
            entity_schema,
            explicit_label=explicit_label,
            explicit_timestamp=explicit_timestamp,
            include_properties=include_properties,
            unix_timestamp=unix_timestamp,
            prefix_id=prefix_id,
        )
        for entity_schema in schema.nodes.values()
    ]
    edges_dfs = [
        pd_from_entity_schema(
            entity_schema,
            explicit_label=explicit_label,
            explicit_timestamp=explicit_timestamp,
            include_properties=include_properties,
            unix_timestamp=unix_timestamp,
            prefix_id=prefix_id,
        )
        for entity_schema in schema.edges.values()
    ]

    nodes_df = pd.concat(nodes_dfs)
    edges_df = pd.concat(edges_dfs)

    nodes_df.sort_values(by=['id'], inplace=True)
    edges_df.sort_values(by=['src', 'dst'], inplace=True)

    # print(nodes_df[nodes_df['id'].duplicated(keep=False)].head(5))
    # print(edges_df[~edges_df.iloc[:, 1].isin(nodes_df.iloc[:, 0])])
    return nodes_df, edges_df
