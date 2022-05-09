import shutil
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from datasets.transforms.define_snapshots import DefineSnapshots
from datasets.transforms.normalize_timestamps import NormalizeTimestamps
from datasets.transforms.sort_edges import SortEdges
from datasets.transforms.undirected import ToUndirected
from datasets.utils.graph_dataset import DATASET_REGISTRY, GraphDataset
from shared.paths import DatasetPath

NUM_CLASSES = 14


def flat_iter(l):
    for el in l:
        if isinstance(el, list):
            yield from flat_iter(el)
        else:
            yield el


@DATASET_REGISTRY
class DBLPHCN(GraphDataset):
    name = 'DBLP-HCN'
    tags = ['dynamic', 'ground-truth', 'overlapping']

    def after_load(self):
        super().after_load()
        self.snapshots = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [
            'node__Author',
            'node__Paper',
            'node__Venue',
            'edge__Author_AUTHORED_Paper',
            'edge__Paper_PUBLISHEDIN_Venue'
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

        # Extract the node labels
        multilabel = torch.zeros([store.num_nodes, NUM_CLASSES], dtype=torch.bool)
        cids = (df['cids'] + np.arange(store.num_nodes) * NUM_CLASSES)
        cids = list(flat_iter(
            cids.apply(lambda x: [] if np.isnan(x).any() else x).apply(list).tolist()
        ))
        multilabel.view(-1)[cids] = True
        store.ground_truth = multilabel
        store.y = multilabel.float().argmax(dim=-1)

        if store._key == 'Paper':
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer('all-MiniLM-L6-v2')
            with torch.no_grad():
                emb = model.encode(df['name'].values, show_progress_bar=True, convert_to_tensor=True).cpu()
            store.x = emb

    def _preprocess(self, data: HeteroData):
        data = super()._preprocess(data)

        self.snapshots = {
            n: DefineSnapshots(n)(data)
            for n in [5, 7]
        }
        torch.save(self.snapshots, self.processed_paths[1])

        return data

    def process(self):
        super().process()

    @staticmethod
    def labels() -> List[str]:
        return ['y', 'louvain', 'label_snapshot_7']
