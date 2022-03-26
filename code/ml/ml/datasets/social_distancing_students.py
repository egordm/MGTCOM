import itertools
from typing import Optional, Callable, Union, List, Tuple
import shutil

import torch

from ml import SortEdges
from ml.datasets.base import GraphDataset, hetero_from_pandas, DATASET_REGISTRY
from ml.transforms.undirected import ToUndirected
from shared.graph.loading import pd_from_entity_schema
from shared.schema import DatasetSchema, GraphSchema


@DATASET_REGISTRY
class SocialDistancingStudents(GraphDataset):
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 model_name: Optional[str] = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def name(self) -> str:
        return 'social-distancing-students'

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['schema.yml']

    def download(self):
        dataset = DatasetSchema.load_schema('social-distancing-student')
        input_dir = dataset.processed()
        shutil.rmtree(self.raw_dir, ignore_errors=True)
        shutil.copytree(input_dir, self.raw_dir, dirs_exist_ok=True)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def process(self):
        from sentence_transformers import SentenceTransformer

        schema = GraphSchema.load_schema(self.raw_dir)

        # Load data
        explicit_label = False
        explicit_timestamp = True
        unix_timestamp = True
        prefix_id = False
        include_properties = lambda cs: [c for c in cs if c.startswith('feat_') or c == 'name' or c == 'text']
        # include_properties = lambda cs: [c for c in cs]

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
                .rename(columns={'id': '_id'})
            for node_type, entity in schema.nodes.items()
        }
        node_mapping = {
            node_type: df['_id'].reset_index(drop=False)
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
                .drop_duplicates(subset=['src', 'dst', 'timestamp'] if entity.dynamic else ['src', 'dst'])
                .merge(node_mapping[entity.source_type], left_on='src', right_on='_id')
                .drop(columns=['src', '_id'])
                .rename(columns={'gid': 'src'})
                .merge(node_mapping[entity.target_type], left_on='dst', right_on='_id')
                .drop(columns=['dst', '_id'])
                .rename(columns={'gid': 'dst'})
                .sort_values(['src', 'dst'], ignore_index=True)
                .rename(columns={'index': 'id'})
            for rel_type, entity in schema.edges.items()
        }

        # Convert BOW features to float
        for node_type, df in node_dfs.items():
            for c in df.columns:
                if not c.startswith('feat_'):
                    continue

                df[c] = df[c].astype(float).fillna(0)

        # Convert timestamps to int
        for entity_type, df in itertools.chain(edge_dfs.items(), node_dfs.items()):
            if 'timestamp' not in df.columns:
                continue
            df['timestamp'] = df['timestamp'].astype(int)

        # Embed natural text features
        model = SentenceTransformer(self.model_name)
        with torch.no_grad():
            df = node_dfs['Tweet']
            emb = model.encode(
                df['text'].values, show_progress_bar=True, convert_to_tensor=True
            ).cpu()
        node_dfs['Tweet'].drop(columns=['text'], inplace=True)

        data = hetero_from_pandas(node_dfs, edge_dfs)
        data['Tweet'].x = emb

        # Normalize timestamps
        timestamps = torch.cat(list(data.timestamp_dict.values()))
        timestamps = timestamps[timestamps != -1]
        min_timestamp = timestamps.double().sort().values.quantile(torch.tensor(0.025).double()).int()
        for entity_type, ts in data.timestamp_dict.items():
            ts[torch.logical_and(ts != -1, ts < min_timestamp)] = min_timestamp
            ts[ts != -1] -= min_timestamp
            data[entity_type].timestamp = ts.long()

        data = ToUndirected(reduce=None)(data)
        data = SortEdges()(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])