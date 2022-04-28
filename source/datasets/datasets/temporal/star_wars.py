import shutil
from typing import List

import torch

from datasets.transforms.define_snapshots import DefineSnapshots
from datasets.transforms.normalize_timestamps import NormalizeTimestamps
from datasets.transforms.sort_edges import SortEdges
from datasets.transforms.undirected import ToUndirected
from datasets.utils.graph_dataset import DATASET_REGISTRY, GraphDataset
from shared.paths import DatasetPath


@DATASET_REGISTRY
class StarWars(GraphDataset):
    name = 'star-wars'
    tags = ['dynamic', 'interaction']

    def after_load(self):
        super().after_load()
        self.snapshots = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [
            'node__Character',
            'edge__Character_INTERACTIONS_Character',
            'edge__Character_MENTIONS_Character'
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

    def _preprocess(self):
        data = self._process_graph(self.raw_paths)

        data = ToUndirected(reduce=None)(data)
        data = SortEdges()(data)
        data = NormalizeTimestamps()(data)

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
        return ['louvain', 'label_snapshot_7']




