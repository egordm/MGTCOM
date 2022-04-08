import shutil

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage

from datasets.transforms.define_snapshots import DefineSnapshots
from datasets.transforms.normalize_timestamps import NormalizeTimestamps
from datasets.transforms.sort_edges import SortEdges
from datasets.transforms.undirected import ToUndirected
from datasets.utils.base import GraphDataset, DATASET_REGISTRY
from shared.paths import DatasetPath


@DATASET_REGISTRY
class ICEWS0515(GraphDataset):
    name = 'icews05-15'
    tags = ['dynamic', 'interaction']

    def after_load(self):
        super().after_load()
        self.snapshots = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [
            'node__Entity',
            'edge__Entity_Rel_Entity',
        ]

    def download(self):
        shutil.rmtree(self.raw_dir, ignore_errors=True)
        shutil.copytree(DatasetPath(self.name).processed_str(), self.raw_dir, dirs_exist_ok=True)

    @property
    def processed_file_names(self):
        return [
            'data.pt',
            'snapshots.pt'
        ]

    def _process_node(self, data: HeteroData, store: NodeStorage, df: pd.DataFrame):
        super()._process_node(data, store, df)

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        with torch.no_grad():
            emb = model.encode(df['name'].values, show_progress_bar=True, convert_to_tensor=True).cpu()
        store.x = emb

    def _process_edge(self, data: HeteroData, store: EdgeStorage, df: pd.DataFrame):
        src, rel, dst = store._key

        if rel == 'Rel':
            del data[store._key]

            rels = df['type'].unique()
            for rel_type in rels:
                store = data[(src, rel_type, dst)]
                edge_df = df[df['type'] == rel_type]
                self._process_edge(data, store, edge_df)

            return

        else:
            super()._process_edge(data, store, df)

            store.train = torch.tensor(df['train'].values.astype(bool), dtype=torch.bool)
            store.valid = torch.tensor(df['valid'].values.astype(bool), dtype=torch.bool)
            store.test = torch.tensor(df['test'].values.astype(bool), dtype=torch.bool)

    def process(self):
        data = self._process_graph(self.raw_paths)

        data = ToUndirected(reduce=None)(data)
        data = SortEdges()(data)
        data = NormalizeTimestamps()(data)

        snapshots = {
            n: DefineSnapshots(n)(data)
            for n in [10, 20]
        }

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        torch.save(snapshots, self.processed_paths[1])
