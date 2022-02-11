from typing import Dict, Tuple, Any
import torchmetrics

import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    _metrics: Dict[str, Tuple[torchmetrics.Metric, bool]]

    def __init__(self) -> None:
        super().__init__()
        metrics = self.configure_metrics()
        self._metrics = {}
        for key, (metric, prog_bar) in metrics.items():
            for prefix in ['train', 'val', 'test']:
                current_metric = (metric if prefix == 'train' else metric.clone(), prog_bar)
                self._metrics[f'{prefix}/{key}'] = current_metric
                setattr(self, f'{prefix}_{key}', current_metric[0])

    def configure_metrics(self) -> Dict[str, Tuple[torchmetrics.Metric, bool]]:
        return {}

    def get_metric(self, key: str, prefix: str = 'train') -> Any:
        return getattr(self, f'{prefix}_{key}')

    def _on_epoch_end(self, prefix: str) -> None:
        for key, (metric, prog_bar) in self._metrics.items():
            if key.startswith(prefix):
                self.log(key, metric.compute(), prog_bar=prog_bar)
                metric.reset()

    def _on_batch_end(self, outputs, prefix: str) -> None:
        for key, (metric, prog_bar) in self._metrics.items():
            if key.startswith(prefix):
                lkey = key.replace(prefix, '')
                if lkey in outputs:
                    if isinstance(outputs[lkey], tuple):
                        metric.update(*outputs[lkey])
                    else:
                        metric.update(outputs[lkey])

    def on_train_batch_end(self, outputs, *args, **kwargs) -> None:
        super().on_train_batch_end(outputs, *args, **kwargs)
        self._on_batch_end(outputs, 'train/')

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self._on_epoch_end('train/')

    def on_validation_batch_end(self, outputs, *args, **kwargs) -> None:
        super().on_validation_batch_end(outputs, *args, **kwargs)
        self._on_batch_end(outputs, 'val/')

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._on_epoch_end('val/')

    def on_test_batch_end(self, outputs, *args, **kwargs) -> None:
        super().on_test_batch_end(outputs, *args, **kwargs)
        self._on_batch_end(outputs, 'test/')

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self._on_epoch_end('test/')


