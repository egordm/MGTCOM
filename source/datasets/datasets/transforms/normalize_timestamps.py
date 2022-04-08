from typing import Any
import itertools as it

import torch
from torch_geometric.transforms import BaseTransform


class NormalizeTimestamps(BaseTransform):
    def __init__(self, p=0.95) -> None:
        super().__init__()
        self.p = p

    def __call__(self, data: Any) -> Any:
        timestamps = torch.cat(
            list(data.timestamp_from_dict.values()) +
            list(data.timestamp_to_dict.values()),
        )
        timestamps = timestamps[timestamps != -1]

        margin = (1.0 - self.p) / 2.0
        min_ts, max_ts = timestamps.double().sort().values\
            .quantile(torch.tensor([margin, 1.0 - margin]).double()).long()

        for entity_type, ts in it.chain(data.timestamp_from_dict.items(), data.timestamp_to_dict.items()):
            ts[torch.logical_and(ts != -1, ts < min_ts)] = min_ts
            ts[torch.logical_and(ts != -1, ts > max_ts)] = max_ts
            ts[ts != -1] -= min_ts

        return data

