from typing import Tuple, List

import igraph as ig
import numpy as np
import pandas as pd
from cdlib.benchmark import RDyn

from shared.graph import DataGraph, CommunityAssignment
from shared.schema import GraphSchema, NodeSchema, GraphProperty, DTypeAtomic, EdgeSchema, DynamicConfig


def generate_rdyn(
        size: int = 300,
        snapshots: int = 5,
        avg_deg: int = 15,
        sigma: float = 0.6,
        lambdad: float = 1,
        alpha: float = 2.5,
        paction: float = 1,
        prenewal: float = 0.8,
        quality_threshold: float = 0.5,
        new_node: float = 0.0,
        del_node: float = 0.0,
        max_evts: int = 1,
        simplified: bool = True,
        **kwargs
) -> List[Tuple[DataGraph, CommunityAssignment]]:
    gen_G, gen_coms = RDyn(
        size=size,
        iterations=snapshots,
        avg_deg=avg_deg,
        sigma=sigma,
        lambdad=lambdad,
        alpha=alpha,
        paction=paction,
        prenewal=prenewal,
        quality_threshold=quality_threshold,
        new_node=new_node,
        del_node=del_node,
        max_evts=max_evts,
        simplified=simplified,
    )

    edges_df = pd.DataFrame([
        {'src': u, 'dst': v, 'timestamp': t_from}
        for (t_from, t_to) in zip(range(snapshots - 1), range(1, snapshots))
        for (u, v) in gen_G.time_slice(t_from, t_to).edges
    ]).sort_values(['src', 'dst', 'timestamp'], ignore_index=True)

    communities = [
        CommunityAssignment.from_clustering(clustering)
        for t, clustering in gen_coms.clusterings.items()
    ]

    schema = GraphSchema() \
        .add_node_schema('Node', NodeSchema(properties={'id': GraphProperty(DTypeAtomic.INT.to_dtype())})) \
        .add_edge_schema('CONNECTED_TO', EdgeSchema(
        dynamic=DynamicConfig('timestamp', True),
        source_type='Node',
        target_type='Node',
        properties={'timestamp': GraphProperty(DTypeAtomic.INT.to_dtype())}))

    G = ig.Graph.DataFrame(
        edges=edges_df,
        directed=False,
    )
    G.es['tstart'] = G.es['timestamp']
    G.es['tend'] = np.array(G.es['timestamp']) + 1

    full_graph = DataGraph(
        schema=schema,
        graph=G
    )

    snapshot_ranges = [(t, t + 1) for t in range(snapshots)]
    graphs = list(map(lambda x: x[1], full_graph.to_snapshots(snapshot_ranges)))

    return list(zip(graphs, communities))
