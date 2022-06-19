import logging
from typing import Any

import torch
from torch_geometric.transforms import BaseTransform


class DefineSnapshots(BaseTransform):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def __call__(self, data: Any) -> Any:
        timestamps = torch.cat(
            list(data.timestamp_from_dict.values()) +
            list(data.timestamp_to_dict.values()),
        )
        timestamps = timestamps[timestamps != -1]

        min_ts = timestamps.min()
        max_ts = timestamps.max()

        if (max_ts - min_ts + 1) < self.n:
            logging.warning(f'Not enough timestamps to define {self.n} snapshots')
            self.n = max_ts - min_ts + 1

        interval = (max_ts - min_ts + 1).float() / self.n
        snapshots = torch.tensor([
            [(min_ts + interval * i).floor(), (min_ts + interval * (i + 1)).ceil()]
            for i in range(self.n)
        ]).long()

        return snapshots

