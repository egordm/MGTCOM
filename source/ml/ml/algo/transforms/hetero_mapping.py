from typing import Dict, Union, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import NodeType, EdgeType

EntityType = Union[NodeType, EdgeType]


class HeteroMappingTransform:
    """
    A transform that maps homogeneous graph entities to heterogenous graph entities. (and vice versa)
    """

    def __init__(self, num_entities_dict: Dict[EntityType, int], to_hetero=True) -> None:
        super().__init__()

        self.num_entities_dict = num_entities_dict
        self.to_hetero = to_hetero

        counter = 0
        self.entity_cumcounts_dict = {}
        for entity_type, num_entities in num_entities_dict.items():
            counter += num_entities
            self.entity_cumcounts_dict[entity_type] = counter

    def __call__(self, *args, **kwargs):
        if self.to_hetero:
            return self.transform(*args, **kwargs)
        else:
            return self.inverse_transform(*args, **kwargs)

    def transform(self, entity_idx: Tensor) -> Tuple[Dict[EntityType, Tensor], Dict[EntityType, Tensor]]:
        entity_idx_dict = {}
        entity_perm_dict = {}

        prev_cumcount = 0
        for entity_type, cumcount in self.entity_cumcounts_dict.items():
            perm = torch.logical_and(entity_idx >= prev_cumcount, entity_idx < cumcount).nonzero().flatten()
            if len(perm) > 0:
                entity_idx_dict[entity_type] = entity_idx[perm] - prev_cumcount
                entity_perm_dict[entity_type] = perm

            prev_cumcount = cumcount

        return entity_idx_dict, entity_perm_dict

    def inverse_transform(self, entity_idx_dict: Dict[EntityType, Tensor],
                          entity_perm_dict: Dict[EntityType, Tensor] = None) -> Tensor:
        if entity_perm_dict is None:
            return torch.cat([
                entity_idx_dict[node_type] + self.entity_cumcounts_dict[node_type]
                for node_type in entity_idx_dict.keys()
            ], dim=0)
        else:
            count = sum([len(perm) for perm in entity_perm_dict.values()])
            entity_idx = torch.zeros(count, dtype=torch.long)
            for node_type, perm in entity_perm_dict.items():
                entity_idx[perm] = entity_idx_dict[node_type] + self.entity_cumcounts_dict[node_type]
            return entity_idx
