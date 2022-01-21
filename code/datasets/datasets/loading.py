from typing import List, Tuple, Iterator, Iterable

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

from datasets.schema import DatasetSchema, NodeSchema, EdgeSchema


def load_nodes(
        schema: NodeSchema,
        include_properties: List[str] = None,
        unix_timestamp: bool = False,
):
    label_prop = schema.get_label()
    timestamp_prop = schema.get_timestamp()

    df = schema.load_df()
    if label_prop:
        df['label'] = df[label_prop.name]
    if timestamp_prop:
        df['timestamp'] = df[timestamp_prop.name]
        if unix_timestamp and str(df['timestamp'].dtype).startswith('datetime'):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.NAN)

    # df['index'] = df['id']
    # df.set_index('index', inplace=True)

    props = {'id', 'label', 'timestamp', 'index', *(include_properties or [])} & set(df.columns)
    df.drop(columns=set(df.columns).difference(props), inplace=True)

    ids = df.pop('id')
    df.insert(0, 'id', ids)

    return df


def load_edges(
        schema: EdgeSchema,
        include_properties: List[str] = None,
        unix_timestamp: bool = False,
):
    timestamp_prop = schema.get_timestamp()

    df = schema.load_df()
    if timestamp_prop:
        df['timestamp'] = df[timestamp_prop.name]
        if unix_timestamp and str(df['timestamp'].dtype).startswith('datetime'):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.NAN)

    props = {'src', 'dst', 'timestamp', *(include_properties or [])} & set(df.columns)
    df.drop(columns=set(df.columns).difference(props), inplace=True)

    dst = df.pop('dst')
    df.insert(0, 'dst', dst)
    src = df.pop('src')
    df.insert(0, 'src', src)

    if not schema.directed:
        df = pd.concat([df, df.rename(columns={'src': 'dst', 'dst': 'src'})])

    return df


def schema_to_pandas_graph(
        schema: DatasetSchema,
        include_properties: List[str] = None,
        prefix_id: bool = False,
        unix_timestamp: bool = False,
):
    nodes_dfs = []
    edges_dfs = []

    for node_schema in schema.nodes:
        df = load_nodes(
            node_schema,
            include_properties=include_properties,
            unix_timestamp=unix_timestamp,
        )
        df['type'] = node_schema.label
        if prefix_id:
            df['id'] = df[['type', 'id']].apply(lambda x: '_'.join(map(str, x)), axis=1)

        nodes_dfs.append(df)

    for edge_schema in schema.edges:
        df = load_edges(
            edge_schema,
            include_properties=include_properties,
            unix_timestamp=unix_timestamp,
        )
        df['type'] = edge_schema.type
        if prefix_id:
            df['src'] = df['src'].apply(lambda x: '_'.join([edge_schema.source, str(x)]))
            df['dst'] = df['dst'].apply(lambda x: '_'.join([edge_schema.target, str(x)]))

        edges_dfs.append(df)

    nodes_df = pd.concat(nodes_dfs)
    edges_df = pd.concat(edges_dfs)

    # print(nodes_df[nodes_df['id'].duplicated(keep=False)].head(5))
    # print(edges_df[~edges_df.iloc[:, 1].isin(nodes_df.iloc[:, 0])])
    return nodes_df, edges_df


def load_igraph(
        schema: DatasetSchema,
        include_properties: List[str] = None,
        directed: bool = False,
        prefix_id: bool = False,
        unix_timestamp: bool = False,
):
    nodes_df, edges_df = schema_to_pandas_graph(
        schema,
        include_properties=include_properties,
        prefix_id=prefix_id,
        unix_timestamp=unix_timestamp,
    )

    return ig.Graph.DataFrame(
        vertices=nodes_df,
        edges=edges_df,
        directed=directed,
    )


def load_nx(
        schema: DatasetSchema,
        include_properties: List[str] = None,
        directed: bool = False,
):
    nodes_df, edges_df = schema_to_pandas_graph(
        schema,
        include_properties=include_properties,
    )
    nodes_df.set_index('id', inplace=True)

    G = nx.from_pandas_edgelist(
        edges_df,
        source='src',
        target='dst',
        edge_attr=True,
        create_using=nx.MultiDiGraph() if directed else nx.MultiGraph(),
    )

    for attr in nodes_df.columns:
        if attr in ['id']:
            continue

        nx.set_node_attributes(G, nodes_df[attr].to_dict(), attr)

    return G


def igraph_to_nx(
        G: ig.Graph,
) -> nx.Graph:
    G_type = nx.MultiDiGraph if G.is_directed() else nx.MultiGraph
    names = G.vs['name']

    edge_df = pd.DataFrame([
        (names[s], names[t])
        for (s, t) in G.get_edgelist()
    ], columns=['src', 'dst'])
    for attr in G.edge_attributes():
        edge_df[attr] = G.es[attr]

    nx_G = nx.from_pandas_edgelist(
        edge_df,
        source='src',
        target='dst',
        edge_attr=True,
        create_using=G_type,
    )

    for attr in G.vertex_attributes():
        attr_data = {k: v for k, v in zip(names, G.vs[attr])}
        nx.set_node_attributes(nx_G, attr_data, attr)

    return nx_G


def tuple_to_dict(data: tuple, keys) -> dict:
    return dict(zip(keys, data))


def write_edgelist(edges: Iterable[Tuple[int, int]], f):
    for edge in sorted(edges):
        f.write('{}\t{}\n'.format(*edge))


def igraph_to_edgelist(graph: ig.Graph):
    if 'gid' not in graph.vs.attributes() or 'gid' not in graph.es.attributes():
        raise ValueError('igraph graph must have gid attributes')

    gids = graph.vs['gid']
    edges = {
        (gids[s] + 1, gids[t] + 1)
        for (s, t) in graph.get_edgelist()
    }

    return edges
