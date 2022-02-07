from typing import List, Any, Tuple

import torch
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader


class EdgeLoader(BaseDataLoader):
    def __init__(
            self,
            data: HeteroData,
            num_samples,
            input_nodes,
            input_edges,
            **kwargs
    ):
        self.hgt_loader = HGTLoader(data, num_samples, input_nodes)

        super(EdgeLoader, self).__init__(
            input_edges.tolist(),
            collate_fn=self.sample,
            **kwargs,
        )

    def sample(self, indices: List[Tuple[int, int, int]]):
        idx_a, idx_b, labels = list(zip(*indices))
        return self.hgt_loader.sample(idx_a), self.hgt_loader.sample(idx_b), torch.tensor(labels, dtype=torch.int64)

    def transform_fn(self, out: Any) -> HeteroData:
        out_a, out_b, labels = out
        return self.hgt_loader.transform_fn(out_a), self.hgt_loader.transform_fn(out_b), labels
