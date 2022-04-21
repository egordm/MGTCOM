from abc import abstractmethod
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


class HeteroConvLayer(torch.nn.Module):
    repr_dim: int

    def forward(self, data: HeteroData, X_dict: Dict[NodeType, Tensor] = None) -> Dict[NodeType, Tensor]:
        return self.convolve(data, X_dict or data.x_dict)

    @abstractmethod
    def convolve(self, data: HeteroData, X_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        raise NotImplementedError

    @staticmethod
    def _process_batch(data: HeteroData, Z_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        # Return node represenations
        batch_size_dict = data.batch_size_dict
        return {
            node_type: Z_dict[node_type][:batch_size]
            for node_type, batch_size in batch_size_dict.items()
        }