from typing import Union

from torch import Tensor

from ml.algo.transforms import HeteroMappingTransform
from ml.data.samplers.ballroom_sampler import BallroomSampler
from ml.data.samplers.base import Sampler
from ml.data.samplers.hgt_sampler import HGTSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecBatch


class HybridSampler(Sampler):
    def __init__(self, n2v_sampler: Union[Node2VecSampler, BallroomSampler], hgt_sampler: HGTSampler) -> None:
        super().__init__()
        self.n2v_sampler = n2v_sampler
        self.hgt_sampler = hgt_sampler
        self.mapper = HeteroMappingTransform(hgt_sampler.data.num_nodes_dict)

    def sample(self, node_ids: Tensor) -> Node2VecBatch:
        pos_walks, neg_walks, node_idx = self.n2v_sampler.sample(node_ids)

        node_idx_dict, node_perm_dict = self.mapper.transform(node_idx)
        data = self.hgt_sampler.sample(node_idx_dict)

        for node_type, node_perm in node_perm_dict.items():
            data[node_type].batch_perm = node_perm

        return Node2VecBatch(pos_walks, neg_walks, data)
