from os import PathLike
from typing import Union, List

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class PretrainedEmbeddingsDataset(Dataset):
    def __init__(self, data: HeteroData, name=None) -> None:
        super().__init__()
        self.data = data
        self.repr_dim = next(store.num_features for store in data.node_stores)
        assert all(store.num_features == self.repr_dim for store in
                   data.node_stores), "All node stores must have the same dimension"

        self.emb = torch.cat(list(data.x_dict.values()), dim=0)
        self.name = name
        # TODO: add possibility to transform indices back to node_type indices

    def __len__(self):
        return self.data.num_nodes

    def __getitem__(self, idx: Union[List[int], Tensor]):
        return self.emb[idx]

    @staticmethod
    def from_pretrained(file: Union[PathLike, str], name=None):
        emb_dict = torch.load(str(file))
        data = HeteroData()
        for node_type, emb in emb_dict.items():
            data[node_type].x = emb

        return PretrainedEmbeddingsDataset(data, name=name)
