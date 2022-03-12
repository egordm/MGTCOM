from typing import Optional, Callable, Union, List, Tuple
import shutil

import torch

from ml.datasets.base import InMemoryDataset, hetero_from_pandas
from shared.graph.loading import pd_from_entity_schema
from shared.schema import DatasetSchema, GraphSchema


class StarWars(InMemoryDataset):
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def name(self) -> str:
        return 'starwars'

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['schema.yml']

    def download(self):
        dataset = DatasetSchema.load_schema('star-wars')
        input_dir = dataset.processed()
        shutil.rmtree(self.raw_dir, ignore_errors=True)
        shutil.copytree(input_dir, self.raw_dir, dirs_exist_ok=True)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def process(self):
        schema = GraphSchema.load_schema(self.raw_dir)

        # Load data
        explicit_label = False
        explicit_timestamp = True
        unix_timestamp = True
        prefix_id = False
        include_properties = lambda cs: [c for c in cs if c.startswith('feat_') or c == 'name']

        # Preprocess node types
        node_dfs = {
            node_type: pd_from_entity_schema(
                entity,
                explicit_label=explicit_label,
                explicit_timestamp=explicit_timestamp,
                include_properties=include_properties,
                unix_timestamp=unix_timestamp,
                prefix_id=prefix_id,
            )
                .set_index('id')
                .drop(columns=['type'])
                .sort_index()
                .reset_index()
                .rename_axis('gid')
            for node_type, entity in schema.nodes.items()
        }
        node_mapping = {
            node_type: df['id']
            for node_type, df in node_dfs.items()
        }

        edge_dfs = {
            (entity.source_type, rel_type, entity.target_type): pd_from_entity_schema(
                entity,
                explicit_label=explicit_label,
                explicit_timestamp=explicit_timestamp,
                include_properties=include_properties,
                unix_timestamp=unix_timestamp,
                prefix_id=prefix_id,
                directed=True,
            )
                .reset_index()
                .drop(columns=['type'])
                .drop_duplicates(subset=['src', 'dst', 'timestamp'])
                .merge(node_mapping[entity.source_type], left_on='src', right_on='id')
                .drop(columns=['src'])
                .rename(columns={'id': 'src'})
                .merge(node_mapping[entity.target_type], left_on='dst', right_on='id')
                .drop(columns=['dst'])
                .rename(columns={'id': 'dst'})
                .sort_values(['src', 'dst'], ignore_index=True)
                .rename(columns={'index': 'id'})
            for rel_type, entity in schema.edges.items()
        }

        data = hetero_from_pandas(node_dfs, edge_dfs)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
