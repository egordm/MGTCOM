from dataclasses import dataclass, field
from typing import List, Dict, Union

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader.utils import filter_hetero_data, to_hetero_csc
from torch_geometric.typing import NodeType
from torch_geometric.loader import HGTLoader

from ml.data.samplers.base import Sampler
from ml.utils import HParams


@dataclass
class HGTSamplerParams(HParams):
    # num_samples: Union[List[int], Dict[NodeType, List[int]]] = field(default_factory=lambda: [3, 2])
    num_samples: List[int] = field(default_factory=lambda: [3, 2])
    """The number of nodes to sample in each iteration and for each node type."""


class HGTSampler(Sampler):
    def __init__(self, data: HeteroData, hparams: HGTSamplerParams = None) -> None:
        super().__init__()
        self.hparams = hparams or HGTSamplerParams()

        self.data = data
        self.num_samples = hparams.num_samples if isinstance(hparams.num_samples, dict) \
            else {key: hparams.num_samples for key in data.node_types}
        self.num_hops = max([len(v) for v in self.num_samples.values()])
        self.sample_fn = torch.ops.torch_sparse.hgt_sample

        self.colptr_dict, self.row_dict, self.perm_dict = to_hetero_csc(data, device='cpu')

    def sample(self, node_ids_dict: Dict[NodeType, Tensor]) -> HeteroData:
        # Correct amount of samples by the batch size
        num_inputs = sum([len(v) for v in node_ids_dict.values()])
        num_samples = {
            k: [n * num_inputs for n in v]
            for k, v in self.num_samples.items()
        }

        node_dict, row_dict, col_dict, edge_dict = self.sample_fn(
            self.colptr_dict,
            self.row_dict,
            node_ids_dict,
            num_samples,
            self.num_hops,
        )

        data = filter_hetero_data(self.data, node_dict, row_dict, col_dict, edge_dict, self.perm_dict)
        for node_type, node_ids in node_ids_dict.items():
            data[node_type].batch_size = len(node_ids)
            data[node_type].node_idx = node_dict[node_type]
            data[node_type].batch_perm = torch.arange(len(node_ids))

        data.batch_size = sum(data.batch_size_dict.values())

        return data
