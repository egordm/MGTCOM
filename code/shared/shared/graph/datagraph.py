from typing import List, Union, Tuple, Iterator

import igraph as ig
import pandas as pd

from datasets.graph_processing import graph_add_timeranges, graph_split_into_snapshots
from shared.graph import igraph_to_edgelist, write_edgelist, EdgeList
from shared.graph.loading import pd_from_graph_schema
from shared.schema import GraphSchema

BaseGraph = ig.Graph


def ide_fix():
    global BaseGraph
    BaseGraph = object


ide_fix()


class DataGraph(BaseGraph):
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

    def add_gids(self):
        self.graph.vs['gid'] = range(self.graph.vcount())
        self.graph.es['gid'] = range(self.graph.ecount())

    def has_gids(self):
        return 'gid' in self.graph.vs.attributes()

    def valid_gids(self):
        return self.has_gids() \
               and max(self.graph.vs['gid']) == self.graph.vcount() - 1 \
               and max(self.graph.es['gid']) == self.graph.ecount() - 1

    def save_edgelist(self, path: str):
        edges = self.to_edgelist()
        write_edgelist(edges, str(path))

    def to_edgelist(self) -> EdgeList:
        if not self.valid_gids():
            self.add_gids()
        return igraph_to_edgelist(self.graph)

    def to_nodemapping(self) -> pd.Series:
        if not self.valid_gids():
            self.add_gids()
        result = pd.Series(self.graph.vs['gid'], index=self.graph.vs['name'], name='gid')
        result.index.name = 'id'
        return result

    def save_nodemapping(self, path: str):
        nodemapping = self.to_nodemapping()
        nodemapping += 1
        nodemapping.to_csv(str(path), sep='\t', index=True, header=True, index_label='id', columns=['gid'])

    def add_timeranges(self):
        graph_add_timeranges(self.schema, self.graph)

    def has_timeranges(self):
        return 'tstart' in self.graph.attributes()

    def to_snapshots(self, snapshot_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, 'DataGraph']]:
        for i, G in graph_split_into_snapshots(self.schema, self.graph, snapshot_ranges):
            yield i, DataGraph(self.schema, G)
