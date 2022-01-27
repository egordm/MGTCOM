import os
from typing import List, Tuple, Iterator, Union

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.visualization import plot_value_distribution_histogram
from shared.logger import get_logger
from shared.pandas import drop_infna
from shared.schema import EntitySchema, GraphSchema

LOG = get_logger(os.path.basename(__file__))


def graph_split_into_snapshots(
        schema: GraphSchema,
        G: ig.Graph,
        snapshot_ranges: List[Tuple[int, int]]
) -> Iterator[Tuple[int, ig.Graph]]:
    for i, (snapshot_start, snapshot_end) in enumerate(snapshot_ranges):
        LOG.debug(f'Creating snapshot {i} - range {snapshot_start} - {snapshot_end}')
        nodes = G.vs.select(tstart_lt=snapshot_end, tend_ge=snapshot_start) if schema.is_node_temporal() else G.vs
        edges = G.es.select(tstart_lt=snapshot_end, tend_ge=snapshot_start) if schema.is_edge_temporal() else G.es
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
    interaction_types = [schema.get_type() for schema in schemas if schema.dynamic and schema.dynamic.interaction]
    interaction_mask = pd.Series(entities['type']).isin(interaction_types)

    entities['tstart'] = timestamps.fillna(np.NINF).values
    tend = timestamps.fillna(np.PINF)
    tend[~interaction_mask] = np.PINF
    entities['tend'] = tend.values

    return entities


def graph_add_timeranges(
        schema: GraphSchema,
        G: ig.Graph
) -> ig.Graph:
    if schema.is_node_temporal():
        LOG.debug(f'Preparing timestamps for nodes')
        graph_entities_add_timestamps(G.vs, list(schema.nodes.values()))
    if schema.is_edge_temporal():
        LOG.debug(f'Preparing timestamps for edges')
        graph_entities_add_timestamps(G.es, list(schema.edges.values()))
    return G


def graph_split_snapshot_ranges(
        schema: GraphSchema,
        G: ig.Graph,
        snapshot_count=5,
        snapshot_coverage=0.95,
        plot=False
):
    LOG.debug('Counting timestamps')
    timestamps_df: pd.Series = None
    if schema.is_node_temporal():
        timestamps_df = pd.Series(G.vs['tstart'])
    if schema.is_edge_temporal():
        timestamps_df = pd.Series(G.es['tstart']) if timestamps_df is None \
            else pd.concat([timestamps_df, pd.Series(G.es['tstart'])])

    timestamps_df = drop_infna(timestamps_df)

    LOG.debug('Creating snapshot ranges')
    # Take the median #snapshot_coverage of the timestamps
    margin = (1.0 - snapshot_coverage) / 2.0
    timestamp_min = timestamps_df.quantile(margin)
    timestamp_max = timestamps_df.quantile(1.0 - margin)
    snapshot_ranges = list(zip(
        np.linspace(timestamp_min, timestamp_max, snapshot_count + 1)[:-1],
        np.linspace(timestamp_min, timestamp_max, snapshot_count + 1)[1:]
    ))

    # Include data within the margins in the first and last snapshots
    snapshot_ranges[0] = (np.NINF, snapshot_ranges[0][1])
    snapshot_ranges[-1] = (snapshot_ranges[-1][0], np.PINF)

    if plot:
        # Count timestamps
        plot_value_distribution_histogram(
            timestamps_df,
            bins=snapshot_count * 2,
            title=f'Timestamps',
            xlabel='Time',
            ylabel='Node+Edge Count',
        )
        for snapshot_start, snapshot_end in snapshot_ranges[:-1]:
            plt.axvline(x=snapshot_end, color='red')

        if isinstance(plot, str):
            plt.savefig(plot)
        else:
            plt.show()

    return snapshot_ranges


def graph_remove_loose_nodes(graph: ig.Graph):
    to_delete_ids = [v.index for v in graph.vs if v.degree() == 0]
    LOG.debug(f'Removing {len(to_delete_ids)} loose nodes')
    graph.delete_vertices(to_delete_ids)


def graph_subsample(graph: ig.Graph, percentage: float) -> ig.Graph:
    mask = np.random.randint(0, 100, graph.vcount()) < (percentage * 100)
    nids = np.where(mask)[0]
    return graph.induced_subgraph(nids)


def graph_filter_connected_components(graph: ig.Graph, min_size: int):
    components = graph.components('weak')
    n_components_to_remove = len([c for c in components if len(c) < min_size])
    to_delete_ids = [
        nid
        for c in components
        if len(c) < min_size
        for nid in c
    ]
    LOG.debug(
        f'(graph_filter_connected_components) Removing {len(to_delete_ids)} nodes and {n_components_to_remove} components')
    graph.delete_vertices(to_delete_ids)


def graph_join_attribute(
        graph: ig.Graph,
        df: pd.Series,
        attribute_name: str,
        left_attribute: str = 'gid',
        group: bool = False,
):
    indexes = graph.vs[left_attribute]
    if not group:
        df = df[~df.index.duplicated(keep='first')]
        values = df.loc[indexes]
    else:
        raise NotImplementedError()
    graph.vs[attribute_name] = values.values
