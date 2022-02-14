from typing import List, Any, Tuple, Union, Dict, Optional, Callable

import torch
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl

from ml.data.splitting import get_edges_partition
from ml.data import LinkSplitter


class EdgeLoader(BaseDataLoader):
    def __init__(
            self,
            data: HeteroData,
            num_neighbors,
            input_nodes,
            input_edges,
            **kwargs
    ):
        self.node_loader = NeighborLoader(
            data=data, num_neighbors=num_neighbors, input_nodes=input_nodes, directed=False, replace=False
        )

        super(EdgeLoader, self).__init__(
            input_edges.tolist(),
            collate_fn=self.sample,
            **kwargs,
        )

    def sample(self, indices: List[Tuple[int, int, int]]):
        idx_a, idx_b, labels = list(zip(*indices))

        return self.node_loader.neighbor_sampler(idx_a), self.node_loader.neighbor_sampler(idx_b), torch.tensor(labels, dtype=torch.int64)

    def transform_fn(self, out: Any) -> HeteroData:
        out_a, out_b, labels = out
        return self.node_loader.transform_fn(out_a), self.node_loader.transform_fn(out_b), labels


class EdgeLoaderDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data: HeteroData,
            node_type: str,
            num_val=0.3,
            num_test=0.0,
            neg_sample_ratio=1.0,
            num_neighbors=None,
            batch_size=8,
            num_workers=0,
            transform=None,
            **kwargs,
    ):
        super().__init__()
        self.data = data
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [4] * 2
        self.num_workers = num_workers
        self.node_type = node_type

        split_transform = LinkSplitter(
            num_val=num_val,
            num_test=num_test,
            edge_types=data.edge_types,
        )

        self.args = {
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'num_neighbors': self.num_neighbors,
        }

        self.train_data, self.val_data, self.test_data = split_transform(data)
        if transform:
            self.train_data, self.val_data, self.test_data = (
                transform(self.train_data), transform(self.val_data), transform(self.test_data)
            )

    def train_dataloader(self):
        data = self.train_data
        edge_index = get_edges_partition(data, partition=0, neg_sample_ratio=self.neg_sample_ratio)
        nodes = (self.node_type, torch.tensor(range(data[self.node_type].num_nodes)))

        return EdgeLoader(
            data,
            **self.args,
            shuffle=True,
            input_nodes=nodes,
            input_edges=torch.cat([*edge_index.values()], dim=1).t(),
        )

    def val_dataloader(self):
        data = self.val_data
        edge_index = get_edges_partition(data, partition=1, neg_sample_ratio=self.neg_sample_ratio)
        nodes = (self.node_type, torch.tensor(range(data[self.node_type].num_nodes)))

        return EdgeLoader(
            self.train_data,
            **self.args,
            input_nodes=nodes,
            input_edges=torch.cat([*edge_index.values()], dim=1).t(),
        )

    def predict_dataloader(self):
        data = self.data
        nodes = (self.node_type, torch.tensor(range(data[self.node_type].num_nodes)))

        return NeighborLoader(
            data,
            **self.args,
            input_nodes=nodes,
            shuffle=False,
        )

    def nodes_dataloader(self):
        data = self.data
        nodes = (self.node_type, torch.tensor(range(data[self.node_type].num_nodes)))
        args = {**self.args}
        del args['batch_size']

        return NeighborLoader(
            data,
            **args,
            batch_size=nodes[1].size(0),
            input_nodes=nodes,
            shuffle=False,
        )
