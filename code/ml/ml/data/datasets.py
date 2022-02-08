from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from shared.constants import CACHE_PATH
from shared.graph.loading import pd_from_entity_schema
from shared.schema import GraphSchema, DatasetSchema


class StarWars(InMemoryDataset):
    def __init__(
            self,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
    ):
        CACHE_PATH.mkdir(parents=True, exist_ok=True)
        super().__init__(str(CACHE_PATH), transform, pre_transform, pre_filter)

        # Load schema
        DATASET = DatasetSchema.load_schema('star-wars')
        schema = GraphSchema.from_dataset(DATASET)

        # Load data
        explicit_label = False
        explicit_timestamp = True
        unix_timestamp = True
        prefix_id = False
        include_properties = lambda cs: [c for c in cs if c.startswith('feat_') or c == 'name']

        nodes_dfs = {
            label: pd_from_entity_schema(
                entity_schema,
                explicit_label=explicit_label,
                explicit_timestamp=explicit_timestamp,
                include_properties=include_properties,
                unix_timestamp=unix_timestamp,
                prefix_id=prefix_id,
            ).set_index('id').drop(columns=['type']).sort_index()
            for label, entity_schema in schema.nodes.items()
        }

        ## Nodes are renumbered into consecutive integer ids
        node_mappings_dfs = {
            label: pd.Series(range(len(df)), index=df.index, name='nid')
            for label, df in nodes_dfs.items()
        }

        edges_dfs = {
            label: pd_from_entity_schema(
                entity_schema,
                explicit_label=explicit_label,
                explicit_timestamp=explicit_timestamp,
                include_properties=include_properties,
                unix_timestamp=unix_timestamp,
                prefix_id=prefix_id,
            )
                .reset_index()
                .drop(columns=['type'])
                .drop_duplicates(subset=['src', 'dst', 'timestamp'])
                .join(node_mappings_dfs[entity_schema.source_type], on='src')
                .drop(columns=['src'])
                .rename(columns={'nid': 'src'})
                .join(node_mappings_dfs[entity_schema.target_type], on='dst')
                .drop(columns=['dst'])
                .rename(columns={'nid': 'dst'})
            for label, entity_schema in schema.edges.items()
        }

        # Create pyg data
        data = HeteroData()
        for ntype, ndf in nodes_dfs.items():
            columns = [c for c in ndf.columns if c.startswith('feat_')]
            data[ntype].x = torch.tensor(ndf[columns].values.astype(np.float32))
            if 'timestamp' in ndf.columns:
                data[ntype].timestamp = torch.tensor(ndf['timestamp'].values.astype(np.int32))

        for etype, edf in edges_dfs.items():
            columns = [c for c in edf.columns if c.startswith('feat_')]
            edge_schema = schema.edges[etype]
            edge_type = (edge_schema.source_type, edge_schema.get_type(), edge_schema.target_type)
            data[edge_type].edge_attr = torch.tensor(edf[columns].values.astype(np.float32))
            data[edge_type].edge_index = torch.tensor(edf[['src', 'dst']].T.values.astype(np.int64))
            if 'timestamp' in edf.columns:
                data[edge_type].timestamp = torch.tensor(edf['timestamp'].values.astype(np.int32))

        self.data, self.slices = self.collate([data])

    def node_mapping(self):
        return list(range(self.data.num_nodes))

