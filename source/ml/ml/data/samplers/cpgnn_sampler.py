from dataclasses import dataclass, field
from typing import Callable, Any, List, Tuple

import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes

from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecBatch, Node2VecSampler, Node2VecSamplerParams
from ml.utils import HParams


@dataclass
class CPGNNSamplerParams(HParams):
    k_length: int = 3
    walk_length: int = 8
    """Length of the random walk."""
    walks_per_node: int = 20
    """Number of random walks to start at each node."""
    num_neg_samples: int = 1
    """The number of negative samples to use for each positive sample."""



class CPGNNSampler(Sampler):
    def __init__(
        self,
        edge_index: Tensor,
        num_nodes=None,
        hparams: CPGNNSamplerParams = None,
        transform_meta: Callable[[Tensor], Any] = None,
    ) -> None:
        super().__init__()

        self.hparams = hparams or CPGNNSamplerParams()

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
        self.transform_meta = transform_meta

        self.n2v = Node2VecSampler(
            edge_index,
            num_nodes,
            Node2VecSamplerParams(
                walk_length=self.hparams.walk_length,
                context_size=(self.hparams.k_length + 1),
                walks_per_node=self.hparams.walks_per_node,
                num_neg_samples=self.hparams.num_neg_samples,
            )
        )

    def sample(self, k: int, node_ids: Tensor) -> Tuple[int, Node2VecBatch]:
        # Update N2V params to sample k long paths
        self.n2v.hparams.context_size = k + 1

        pos_walks, neg_walks, node_idx = self.n2v.sample(node_ids)

        pos_walks = torch.stack([pos_walks[:, 0], pos_walks[:, -1]], dim=1)
        neg_walks = torch.stack([neg_walks[:, 0], neg_walks[:, -1]], dim=1)
        walks = torch.cat([pos_walks, neg_walks], dim=0).view(-1)
        walks = node_idx[walks]

        node_idx, perm = torch.unique(walks.view(-1), return_inverse=True)
        walks = perm.view(-1, 2)
        pos_walks, neg_walks = walks[:pos_walks.shape[0]], walks[pos_walks.shape[0]:]

        node_meta = node_idx if self.transform_meta is None else self.transform_meta(node_idx)
        return k, Node2VecBatch(pos_walks, neg_walks, node_meta)

