from typing import List, Any, Tuple, Union, Dict, Optional, Callable

import torch
from torch import Tensor
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from torch_geometric.typing import NodeType
from torch_geometric.utils import negative_sampling
import pytorch_lightning as pl

from ml.data import LinkSplitter


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
        # TODO: Modify transform_fn of hgt loader to set batch_size for every node type given we sample edges (not
        #  nodes), also do the same for node sampling
        return self.hgt_loader.transform_fn(out_a), self.hgt_loader.transform_fn(out_b), labels


class HGTLoaderWrapper(HGTLoader):
    def __init__(
            self,
            data: HeteroData,
            num_samples: Union[List[int], Dict[NodeType, List[int]]],
            input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
            transform: Callable = None,
            dataset=None,
            **kwargs
    ):
        self.data = data
        self.num_samples = num_samples
        self.input_nodes = input_nodes
        super().__init__(data, num_samples, input_nodes, transform, **kwargs)


class EdgeLoaderDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data: HeteroData,
            node_type: str,
            num_val=0.3,
            num_test=0.0,
            neg_sample_ratio=1.0,
            num_samples=None,
            batch_size=8,
            num_workers=0,
            **kwargs,
    ):
        super().__init__()
        self.data = data
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples or [4] * 2
        self.num_workers = num_workers
        self.node_type = node_type

        transform = LinkSplitter(
            num_val=num_val,
            num_test=num_test,
            edge_types=data.edge_types,
        )

        self.train_data, self.val_data, self.test_data = transform(data)

    def _get_edges_partition(self, data: HeteroData, partition: int) -> torch.Tensor:
        pos_edge_index = {
            edge_type: data[edge_type].edge_index[:, data[edge_type].edge_partitions == partition]
            for edge_type in data.edge_types
        }

        neg_edge_index = {
            edge_type: negative_sampling(
                data[edge_type].edge_index,
                num_neg_samples=int(pos_edge_index[edge_type].shape[1] * self.neg_sample_ratio)
            )
            for edge_type in data.edge_types
        }

        edge_index = {
            edge_type: torch.cat([
                torch.cat(
                    [pos_edge_index[edge_type], torch.ones(1, pos_edge_index[edge_type].shape[1], dtype=torch.long)],
                    dim=0),
                torch.cat(
                    [neg_edge_index[edge_type], torch.zeros(1, neg_edge_index[edge_type].shape[1], dtype=torch.long)],
                    dim=0),
            ], dim=1)
            for edge_type in data.edge_types
        }

        return edge_index

    def train_dataloader(self):
        data = self.train_data
        edge_index = self._get_edges_partition(data, partition=0)
        nodes = (self.node_type, torch.tensor(range(data[self.node_type].num_nodes)))

        return EdgeLoader(
            data,
            num_samples=self.num_samples,
            shuffle=True,
            input_nodes=nodes,
            input_edges=torch.cat([*edge_index.values()], dim=1).t(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        data = self.val_data
        edge_index = self._get_edges_partition(data, partition=1)
        nodes = (self.node_type, torch.tensor(range(data[self.node_type].num_nodes)))

        return EdgeLoader(
            self.train_data,
            num_samples=self.num_samples,
            input_nodes=nodes,
            input_edges=torch.cat([*edge_index.values()], dim=1).t(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        data = self.data
        nodes = (self.node_type, torch.tensor(range(data[self.node_type].num_nodes)))

        return HGTLoaderWrapper(
            data,
            num_samples=self.num_samples,
            input_nodes=nodes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
