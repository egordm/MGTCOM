from ml.data import LinkSplitter
from shared.schema import DatasetSchema, GraphSchema
from shared.graph.loading import pd_from_entity_schema
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData, Data

from shared.schema import DatasetSchema, GraphSchema
from shared.graph.loading import pd_from_entity_schema

DATASET = DatasetSchema.load_schema('star-wars')
schema = GraphSchema.from_dataset(DATASET)

explicit_label = False
explicit_timestamp = True
unix_timestamp = True
prefix_id = None
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

cursor = 0
for df in edges_dfs.values():
    df.index += cursor
    cursor += len(df)

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


metapath = [
    ('Character', 'INTERACTIONS', 'Character'),
    ('Character', 'MENTIONS', 'Character')
]

transform = LinkSplitter(
    edge_types=metapath,
)

train_data, val_data, test_data = transform(data)

u = 0