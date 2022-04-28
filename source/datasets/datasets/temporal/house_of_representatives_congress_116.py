import shutil
from typing import List

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


@DATASET_REGISTRY
class HouseOfRepresentativesCongress116(GraphDataset):
    name = 'house-of-representatives-congress-116'
    tags = ['dynamic', 'interaction']

    def after_load(self):
        super().after_load()
        self.snapshots = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [
            'node__Member',
            'node__Commitee',
            'node__Bill',
            'node__Rollcall',
            'edge__Member_SPONSORS_Bill',
            'edge__Member_COSPONSORS_Bill',
            'edge__Commitee_COMMITEES_Bill',
            'edge__Member_ISIN_Commitee',
            'edge__Bill_RELATEDTO_Bill',
            'edge__Rollcall_ROLLCALLFOR_Bill',
            'edge__Member_VOTEDFOR_Rollcall',
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

        # if store._key == 'Commitee' or store._key == 'Bill' or store._key == 'Rollcall':
        if store._key == 'Commitee':
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer('all-MiniLM-L6-v2')
            with torch.no_grad():
                emb = model.encode(df['name'].values, show_progress_bar=True, convert_to_tensor=True).cpu()

            store.x = emb

    def _preprocess(self):
        data = self._process_graph(self.raw_paths)

        data = ToUndirected(reduce=None)(data)
        data = SortEdges()(data)
        data = NormalizeTimestamps()(data)

        self.snapshots = {
            n: DefineSnapshots(n)(data)
            for n in [5, 8]
        }

        torch.save(self.snapshots, self.processed_paths[1])

        return data

    def process(self):
        super().process()

    @staticmethod
    def labels() -> List[str]:
        return ['louvain', 'label_snapshot_8']
