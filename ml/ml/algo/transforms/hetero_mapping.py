from typing import Dict, Union, Tuple, NamedTuple, List

import torch
from torch import Tensor
from torch_geometric.typing import NodeType, EdgeType

EntityType = Union[NodeType, EdgeType]


class ToHeteroMappingTransform:
    """
    A transform that maps homogeneous graph entities to heterogenous graph entities. (and vice versa)
    """

    def __init__(
            self,
            num_entities_dict: Dict[EntityType, int]
    ) -> None:
        super().__init__()

        self.num_entities_dict = num_entities_dict

        offset = 0
        self.entity_range_dict = {}
        for entity_type, num_entities in num_entities_dict.items():
            self.entity_range_dict[entity_type] = (offset, offset + num_entities)
            offset += num_entities

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, entity_idx: Tensor) -> Tuple[Dict[EntityType, Tensor], Dict[EntityType, Tensor]]:
        entity_idx_dict = {}
        entity_perm_dict = {}

        for entity_type, (offset_from, offset_to) in self.entity_range_dict.items():
            perm = torch.logical_and(entity_idx >= offset_from, entity_idx < offset_to).nonzero().flatten()
            if len(perm) > 0:
                entity_idx_dict[entity_type] = entity_idx[perm] - offset_from
                entity_perm_dict[entity_type] = perm

        return entity_idx_dict, entity_perm_dict

    def inverse_transform(
            self,
            entity_idx_dict: Dict[EntityType, Tensor],
            entity_perm_dict: Dict[EntityType, Tensor] = None
    ) -> Tensor:
        if entity_perm_dict is None:
            entity_idx = torch.cat([
                entity_idx + self.entity_range_dict[entity_type][0]
                for entity_type, entity_idx in entity_idx_dict.items()
            ], dim=0)
        else:
            count = sum(map(len, entity_perm_dict.values()))
            entity_idx = torch.zeros(count, dtype=torch.long)
            for entity_type, perm in entity_perm_dict.items():
                entity_idx[perm] = entity_idx_dict[entity_type] + self.entity_range_dict[entity_type][0]

        return entity_idx

    @staticmethod
    def inverse_transform_values(
            entity_value_dict: Dict[EntityType, Tensor],
            entity_perm_dict: Dict[EntityType, Tensor] = None,
            shape: List[int] = None,
            device: torch.device = None,
            dtype: torch.dtype = None,
    ):
        if entity_perm_dict is None:
            entity_value = torch.cat(list(entity_value_dict.values()), dim=0)
        else:
            count = sum(map(len, entity_perm_dict.values()))
            entity_value = torch.zeros(count, *shape, dtype=dtype, device=device)
            for entity_type, perm in entity_perm_dict.items():
                entity_value[perm] = entity_value_dict[entity_type]

        return entity_value
