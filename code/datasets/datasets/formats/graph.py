import igraph as ig
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
