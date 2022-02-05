import os
import pathlib
from typing import List, Union, Tuple, Iterator, Optional, Callable

import igraph as ig
import numpy as np
import pandas as pd
import yaml

from datasets.graph_processing import graph_add_timeranges, graph_split_into_snapshots, graph_subsample, \
    graph_filter_connected_components
from shared.graph import igraph_to_edgelist, write_edgelist, EdgeList
from shared.graph.loading import pd_from_graph_schema
from shared.logger import get_logger
from shared.schema import GraphSchema

LOG = get_logger(os.path.basename(__file__))

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
            include_properties: Optional[Union[List[str], Callable[[List[str]], List[str]]]] = None,
            prefix_id: bool = False,
            explicit_timestamp: bool = True,
            unix_timestamp: bool = False,
    ) -> Union[ig.Graph, 'DataGraph']:
        nodes_df, edges_df = pd_from_graph_schema(
            schema,
            include_properties=include_properties,
            prefix_id=prefix_id,
            explicit_timestamp=explicit_timestamp,
            unix_timestamp=unix_timestamp,
        )

        graph = ig.Graph.DataFrame(
            vertices=nodes_df,
            edges=edges_df,
            directed=True,
        )

        return DataGraph(schema, graph)

    def add_gids(self):
        LOG.debug('Renumbering nodes gid')
        if 'tstart' in self.graph.vs.attributes():
            v_order = np.argsort(self.graph.vs['tstart'])
        else:
            v_order = np.arange(self.graph.vcount())

        v_indexed = np.dstack((v_order, np.arange(self.graph.vcount())))[0]
        v_indexed = v_indexed[np.argsort(v_indexed[:, 0])]

        self.graph.vs['gid'] = v_indexed[:, 1]
        self.graph.es['gid'] = range(self.graph.ecount())

    def has_gids(self):
        return 'gid' in self.graph.vs.attributes()

    def valid_gids(self):
        return self.has_gids() \
               and max(self.graph.vs['gid']) == self.graph.vcount() - 1 \
               and max(self.graph.es['gid']) == self.graph.ecount() - 1

    def has_gids(self):
        if 'gid' in self.graph.vs.attributes() and 'gid' in self.graph.es.attributes():
            return True
        return False

    def save_edgelist(self, path: str):
        edges = self.to_edgelist()
        write_edgelist(edges, str(path))
        self.write_graph_info(str(pathlib.Path(str(path)).with_suffix('.info.yaml')))

    def to_edgelist(self) -> EdgeList:
        if not self.has_gids():
            raise ValueError('Graph has no gids')

        return igraph_to_edgelist(self.graph)

    def to_nodemapping(self) -> pd.Series:
        if not self.has_gids():
            raise ValueError('Graph has no gids')

        result = pd.Series(self.graph.vs['gid'], index=self.graph.vs['name'], name='gid')
        result.index.name = 'id'
        return result

    def save_nodemapping(self, path: str):
        nodemapping = self.to_nodemapping()
        nodemapping.to_csv(str(path), sep='\t', index=True, header=True, index_label='id', columns=['gid'])

    def add_timeranges(self):
        graph_add_timeranges(self.schema, self.graph)

    def has_timeranges(self):
        return 'tstart' in self.graph.attributes()

    def to_snapshots(self, snapshot_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, 'DataGraph']]:
        for i, G in graph_split_into_snapshots(self.schema, self.graph, snapshot_ranges):
            yield i, DataGraph(self.schema, G)

    def write_graph_info(self, path: str):
        data = {
            'nodes': self.vcount(),
            'edges': self.ecount(),
        }
        with open(str(path), 'w') as f:
            yaml.dump(data, f)

    def subsample_nodes(self, percentage: float) -> 'DataGraph':
        return DataGraph(self.schema, graph_subsample(self.graph, percentage))

    def filter_connected_components(self, min_size: int = 1):
        graph_filter_connected_components(self.graph, min_size)
