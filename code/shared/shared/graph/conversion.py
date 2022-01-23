import igraph as ig
import networkx as nx
import pandas as pd


def igraph_to_edgelist(graph: ig.Graph):
    if 'gid' not in graph.vs.attributes() or 'gid' not in graph.es.attributes():
        raise ValueError('igraph graph must have gid attributes')

    gids = graph.vs['gid']
    edges = {
        (gids[s], gids[t])
        for (s, t) in graph.get_edgelist()
    }
    if not graph.is_directed():
        edges = edges.union({(t, s) for (s, t) in edges})

    df = pd.DataFrame(edges, columns=['src', 'dst']).sort_values(by=['src', 'dst'])
    return df


def igraph_to_nx(graph: ig.Graph, create_using=None):
    names = graph.vs['name']
    edge_df = pd.DataFrame([
        (names[s], names[t])
        for (s, t) in graph.get_edgelist()
    ], columns=['src', 'dst'])
    for attr in graph.edge_attributes():
        edge_df[attr] = graph.es[attr]

    nx_G = nx.from_pandas_edgelist(
        edge_df,
        source='src',
        target='dst',
        edge_attr=True if len(graph.edge_attributes()) > 0 else None,
        create_using=nx.MultiDiGraph if graph.is_directed() else nx.MultiGraph
    )

    for attr in graph.vertex_attributes():
        attr_data = {k: v for k, v in zip(names, graph.vs[attr])}
        nx.set_node_attributes(nx_G, attr_data, attr)

    return nx_G