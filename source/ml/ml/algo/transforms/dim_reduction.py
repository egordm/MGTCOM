from enum import Enum
from typing import Union

import torch
from matplotlib.transforms import IdentityTransform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor
from umap import UMAP

from ml.utils import Metric


class DimensionReductionMode(Enum):
    PCA = "PCA"
    TSNE = "TSNE"
    UMAP = "UMAP"
    Identity = "Identity"


class DimensionReductionTransform:
    mapper: Union[PCA, TSNE, UMAP, IdentityTransform]

    def __init__(
            self,
            n_components=2,
            mode: DimensionReductionMode = DimensionReductionMode.TSNE,
            metric: Metric = Metric.DOTP
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.metric = metric
        self.mode = mode

        if mode == DimensionReductionMode.PCA:
            self.mapper = PCA(n_components=n_components)
        elif mode == DimensionReductionMode.TSNE:
            self.mapper = TSNE(
                n_components=n_components, metric=metric.sk_metric(),
                learning_rate=200.0, init='random'
            )
        elif mode == DimensionReductionMode.UMAP:
            self.mapper = UMAP(
                n_components=2,
                metric="euclidean" if metric == Metric.L2 else "cosine"
            )
        else:
            self.mapper = IdentityTransform()

        self.is_fitted = False

    def fit(self, X: Tensor):
        if self.mode == DimensionReductionMode.TSNE:
            pass
        else:
            self.mapper.fit(X.numpy())
        self.is_fitted = True
        return self

    def transform(self, X: Tensor) -> Tensor:
        if self.mode == DimensionReductionMode.TSNE:
            return torch.from_numpy(self.mapper.fit_transform(X))
        else:
            return torch.from_numpy(self.mapper.transform(X))

    def inverse_transform(self, X_t: Tensor) -> Tensor:
        return torch.from_numpy(self.mapper.inverse_transform(X_t))


class IdentityTransform:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def fit(self, X: Tensor):
        return self

    def transform(self, X: Tensor) -> Tensor:
        return X

    def inverse_transform(self, X_t: Tensor) -> Tensor:
        return X_t
