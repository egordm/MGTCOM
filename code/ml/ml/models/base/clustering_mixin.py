from dataclasses import dataclass
from typing import Any, Union, List

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from ml.algo.clustering import KMeans
from ml.utils import HParams, Metric
from ml.utils.outputs import OutputExtractor


@dataclass
class ClusteringMixinParams(HParams):
    infer_k: int = 20
    """Number of clusters"""


class ClusteringMixin:
    train_outputs: OutputExtractor
    val_outputs: OutputExtractor
    test_outputs: OutputExtractor

    hparams: Union[ClusteringMixinParams, Any]
    repr_dim: int

    mus = None

    def on_train_cluster(self):
        if self.hparams.infer_k > 0:
            if 'Z' in self.train_outputs:
                Z = self.train_outputs.extract_cat('Z', cache=False, device='cpu')
            else:
                Z = self.train_outputs.extract_cat_kv('Z_dict', cache=False, device='cpu')

            kmeans = KMeans(self.repr_dim, k=self.hparams.infer_k, metric=Metric(self.hparams.metric), niter=30)
            kmeans.fit(Z)
            self.mus = torch.from_numpy(kmeans.centroids).float()

    def on_val_cluster_assign(self):
        if self.mus is not None:
            if 'Z' in self.val_outputs:
                X = self.val_outputs.extract_cat('Z', cache=False, device='cpu')
            else:
                X = self.val_outputs.extract_cat_kv('Z_dict', cache=False, device='cpu')

            r = Metric(self.hparams.metric).pairwise_sim_fn(X.unsqueeze(1), self.mus.unsqueeze(0))
            z = r.argmax(dim=-1)
            self.val_outputs.outputs.append({
                'z': z,
                'X': X,
            })

    def on_test_cluster_assign(self):
        if self.mus is not None:
            if 'Z' in self.test_outputs:
                X = self.test_outputs.extract_cat('Z', cache=False, device='cpu')
            else:
                X = self.test_outputs.extract_cat_kv('Z_dict', cache=False, device='cpu')

            r = Metric(self.hparams.metric).pairwise_sim_fn(X.unsqueeze(1), self.mus.unsqueeze(0))
            z = r.argmax(dim=-1)
            self.test_outputs.outputs.append({
                'z': z,
                'X': X,
            })

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)
        self.on_train_cluster()

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().validation_epoch_end(outputs)
        self.on_val_cluster_assign()

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().test_epoch_end(outputs)
        self.on_test_cluster_assign()

    def get_extra_state(self) -> Any:
        return {
            'mus': self.mus,
        }

    def set_extra_state(self, state: Any):
        self.mus = state['mus']