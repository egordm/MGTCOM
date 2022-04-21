from typing import Callable, Dict

import torch
from torch_geometric.typing import NodeType

from ml.algo.transforms.hetero_mapping import ToHeteroMappingTransform
from ml.data.loaders.base import BatchedLoader
from ml.data.transforms.compose import Compose


class NodesLoader(BatchedLoader):
    def __init__(
            self,
            num_nodes: int,
            transform: Callable = None,
            batch_size_tmp: int = None,
            *args, **kwargs
    ):
        kwargs.pop('dataset', None)
        super().__init__(torch.arange(num_nodes), transform, batch_size_tmp=batch_size_tmp, *args, **kwargs)
        self.num_nodes = num_nodes


class HeteroNodesLoader(NodesLoader):
    def __init__(
            self,
            num_nodes_dict: Dict[NodeType, int],
            transform_nodes_fn: Callable = None,
            batch_size_tmp: int = None,
            *args, **kwargs
    ):
        kwargs.pop('transform', None)
        kwargs.pop('dataset', None)
        transform_hetero = ToHeteroMappingTransform(num_nodes_dict)

        def transform_aux(batch):
            node_meta_dict, node_perm_dict = batch
            node_meta_new_dict = transform_nodes_fn(node_meta_dict)
            return node_meta_new_dict, node_perm_dict

        transform = Compose([transform_hetero, transform_aux]) if transform_nodes_fn is not None else transform_hetero
        num_nodes = sum(num_nodes_dict.values())
        super().__init__(num_nodes, transform, batch_size_tmp=batch_size_tmp, *args, **kwargs)
        self.num_nodes_dict = num_nodes_dict
        self.transform_nodes_fn = transform_nodes_fn


