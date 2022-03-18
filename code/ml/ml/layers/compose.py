from typing import Dict

import torch.nn


class Compose(torch.nn.Module):
    def __init__(self, modules: Dict[str, torch.nn.Module], agg='cat') -> None:
        super().__init__()
        self.modules = torch.nn.ModuleDict(modules)
        self.agg = agg

    def forward(self, *args, **kwargs):
        res = []
        for name, module in self.modules.items():
            res.append(module(*args, **kwargs))

        if self.agg == 'cat':
            return torch.cat(res, dim=-1)
        elif self.agg == 'sum':
            return sum(res)
        else:
            raise ValueError(f'Unknown aggregation {self.agg}')
