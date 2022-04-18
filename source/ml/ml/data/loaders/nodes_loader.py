from typing import Callable, Dict

import torch
from torch_geometric.typing import NodeType

from ml.algo.transforms.hetero_mapping import HeteroMappingTransform
from ml.data.loaders.base import BatchedLoader
from ml.data.transforms.compose import Compose


class NodesLoader(BatchedLoader):
    def __init__(self, num_nodes: int, transform: Callable = None, *args, **kwargs):
        super().__init__(torch.arange(num_nodes), transform, *args, **kwargs)
        self.num_nodes = num_nodes


class HeteroNodesLoader(NodesLoader):
    def __init__(self, num_nodes_dict: Dict[NodeType, int], transform: Callable = None, *args, **kwargs):
        transform_hetero = HeteroMappingTransform(num_nodes_dict, to_hetero=True)

        self.transform_nodes_fn = transform

        def transform_aux(batch):
            node_idx_dict, node_perm_dict = batch
            data = self.transform_nodes_fn(node_idx_dict)
            for node_type, node_perm in node_perm_dict.items():
                data[node_type].batch_perm = node_perm

            return data

        transform = Compose([transform_hetero, transform_aux]) if self.transform_nodes_fn is not None else transform_hetero
        num_nodes = sum(num_nodes_dict.values())
        super().__init__(num_nodes, transform, *args, **kwargs)
        self.num_nodes_dict = num_nodes_dict

