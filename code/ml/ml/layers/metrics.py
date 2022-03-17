from typing import Any, Dict

import torch.nn
import torchmetrics


class MetricBag(torch.nn.Module):
    def __init__(self, metrics: Dict[str, torchmetrics.Metric]):
        super().__init__()
        self.metrics_ = torch.nn.ModuleList(metrics.values())
        self.metrics = metrics

    def update(self, inputs: Dict[str, Any]):
        for k, v in inputs.items():
            metric = self.metrics.get(k)
            if metric is not None:
                metric.update(v)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def compute(self, epoch=False, prefix=None):
        outputs = {
            f'{prefix}{k}' if prefix else k: v.compute()
            for k, v in self.metrics.items() if v._update_called
        }

        if epoch:
            self.reset()

        return outputs
