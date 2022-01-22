import os
from typing import List, Tuple, Iterator, Union

import igraph as ig
import numpy as np
import pandas as pd

from datasets.schema import DatasetSchema, EntitySchema
from shared.logger import get_logger

LOG = get_logger(os.path.basename(__file__))


def graph_split_into_snapshots(
        schema: DatasetSchema,
        G: ig.Graph,
        snapshot_ranges: List[Tuple[int, int]]
) -> Iterator[Tuple[int, ig.Graph]]:
    for i, (snapshot_start, snapshot_end) in enumerate(snapshot_ranges):
        LOG.debug(f'Creating snapshot {i} - range {snapshot_start} - {snapshot_end}')
        nodes = G.vs.select(tstart_le=snapshot_end, tend_ge=snapshot_start) if schema.is_node_temporal() else G.vs
        edges = G.es.select(tstart_le=snapshot_end, tend_ge=snapshot_start) if schema.is_edge_temporal() else G.es
        G_snapshot = G.induced_subgraph(nodes).subgraph_edges(edges)

        yield i, G_snapshot


def graph_remove_edges_of_type(G: ig.Graph, edge_types: List[str]) -> ig.Graph:
    return G.subgraph_edges(G.es.select(type_ne=edge_types))


def graph_remove_nodes_of_type(G: ig.Graph, node_types: List[str]) -> ig.Graph:
    return G.induced_subgraph(G.vs.select(type_ne=node_types))


def graph_entities_add_timestamps(
        entities: Union[ig.VertexSeq, ig.EdgeSeq],
        schemas: List[EntitySchema],
) -> Union[ig.VertexSeq, ig.EdgeSeq]:
    timestamps = pd.Series(entities['timestamp'])
    interaction_types = [schema.get_type() for schema in schemas if schema.interaction]
    interaction_mask = pd.Series(entities['type']).isin(interaction_types)

    entities['tstart'] = timestamps.fillna(np.NINF).values
    tend = timestamps.fillna(np.PINF)
    tend[~interaction_mask] = np.PINF
    entities['tend'] = tend.values

    return entities


def graph_add_timeranges(
        schema: DatasetSchema,
        G: ig.Graph
) -> ig.Graph:
    if schema.is_node_temporal():
        LOG.debug(f'Preparing timestamps for nodes')
        graph_entities_add_timestamps(G.vs, schema.nodes)
    if schema.is_edge_temporal():
        LOG.debug(f'Preparing timestamps for edges')
        graph_entities_add_timestamps(G.es, schema.edges)
    return G
