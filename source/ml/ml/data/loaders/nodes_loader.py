from typing import Union, Sized, Callable, Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.transforms import Compose
from torch_geometric.typing import NodeType

from ml.algo.transforms.hetero_mapping import HeteroMappingTransform
from ml.data.loaders.base import BatchedLoader


class NodesLoader(BatchedLoader):
    def __init__(self, num_nodes: int, transform: Callable = None, *args, **kwargs):
        super().__init__(torch.arange(num_nodes), transform, *args, **kwargs)
        self.num_nodes = num_nodes


class HeteroNodesLoader(NodesLoader):
    def __init__(self, num_nodes_dict: Dict[NodeType, int], transform: Callable = None, *args, **kwargs):
        transform_hetero = HeteroMappingTransform(num_nodes_dict, to_hetero=True)
        transform = Compose([transform_hetero, transform]) if transform is not None else transform_hetero
        num_nodes = sum(num_nodes_dict.values())
        super().__init__(num_nodes, transform, *args, **kwargs)
        self.num_nodes_dict = num_nodes_dict

