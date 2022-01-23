from typing import List, Union

import igraph as ig
import networkx as nx
import pandas as pd

from shared.graph.loading import pd_from_graph_schema
from shared.schema import GraphSchema


class DataGraph:
    graph: ig.Graph
    schema: GraphSchema

    def __init__(self, schema: GraphSchema, graph: ig.Graph):
        self.schema = schema
        self.graph = graph

    def __getattr__(self, attr):
        return getattr(self.graph, attr)

    @classmethod
    def from_schema(
            cls,
            schema: GraphSchema,
            include_properties: List[str] = None,
            prefix_id: bool = False,
            unix_timestamp: bool = False,
    ) -> Union[ig.Graph, 'DataGraph']:
        nodes_df, edges_df = pd_from_graph_schema(
            schema,
            include_properties=include_properties,
            prefix_id=prefix_id,
            unix_timestamp=unix_timestamp,
        )

        graph = ig.Graph.DataFrame(
            vertices=nodes_df,
            edges=edges_df,
            directed=True,
        )

        return DataGraph(schema, graph)

