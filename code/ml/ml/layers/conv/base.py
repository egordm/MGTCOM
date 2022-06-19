from abc import abstractmethod
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


class HeteroConvLayer(torch.nn.Module):
    repr_dim: int

    def forward(
        self,
        data: HeteroData,
        X_dict: Dict[NodeType, Tensor] = None,
        return_raw=False,
        *args,
        **kwargs
    ) -> Dict[NodeType, Tensor]:
        Z_dict = self.convolve(data, X_dict or data.x_dict, *args, **kwargs)

        if return_raw:
            return Z_dict
        else:
            return self.process_batch(data, Z_dict)

    @abstractmethod
    def convolve(self, data: HeteroData, X_dict: Dict[NodeType, Tensor], *args, **kwargs) -> Dict[NodeType, Tensor]:
        raise NotImplementedError

    @staticmethod
    def process_batch(data: HeteroData, Z_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        # Return node represenations
        batch_size_dict = data.batch_size_dict
        return {
            node_type: Z_dict[node_type][:batch_size]
            for node_type, batch_size in batch_size_dict.items()
        }
