from collections import defaultdict
from typing import Dict, List

import torch.nn

from ml.utils import merge_dicts


class Compose(torch.nn.Module):
    def __init__(self, modules: Dict[str, torch.nn.Module], agg='cat') -> None:
        super().__init__()
        self.modules_dict = torch.nn.ModuleDict(modules)
        self.agg = agg

        if self.agg == 'cat':
            self.agg_fn = lambda xs: torch.cat(xs, dim=-1)
        elif self.agg == 'sum':
            self.agg_fn = sum
        else:
            raise ValueError(f'Unknown aggregation {self.agg}')

    def compose_tensor(self, out: List[torch.Tensor]) -> torch.Tensor:
        return self.agg_fn(out)

    def compose_dict(self, out: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return merge_dicts(out, self.compose_tensor)

    def forward(self, *args, **kwargs):
        res = []
        for name, module in self.modules_dict.items():
            res.append(module(*args, **kwargs))

        if isinstance(res[0], torch.Tensor):
            return self.compose_tensor(res)
        else:
            return self.compose_dict(res)
