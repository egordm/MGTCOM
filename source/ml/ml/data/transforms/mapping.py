from typing import Union

import torch
import umap

from sklearn.decomposition import PCA
from torch import Tensor


class PCAMapper:
    def __init__(self, n_components=2) -> None:
        super().__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=2)

    def fit(self, X: Tensor):
        self.pca.fit(X.numpy())
        return self

    def transform(self, X: Tensor) -> Tensor:
        return torch.from_numpy(self.pca.transform(X))

    def inverse_transform(self, X_t: Tensor) -> Tensor:
        return torch.from_numpy(self.pca.inverse_transform(X_t))


class UMAPMapper:
    def __init__(self, n_components=2) -> None:
        super().__init__()
        self.n_components = n_components
        self.umap = umap.UMAP(n_components=2)

    def fit(self, X: Tensor):
        self.umap.fit(X.numpy())
        return self

    def transform(self, X: Tensor) -> Tensor:
        return torch.from_numpy(self.umap.transform(X))

    def inverse_transform(self, X_t: Tensor) -> Tensor:
        return torch.from_numpy(self.umap.inverse_transform(X_t))


class Identity:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def fit(self, X: Tensor):
        return self

    def transform(self, X: Tensor) -> Tensor:
        return X

    def inverse_transform(self, X_t: Tensor) -> Tensor:
        return X_t


def mapper_cls(name: str) -> type:
    if name == 'pca':
        return PCAMapper
    elif name == 'umap':
        return UMAPMapper
    elif name == 'none':
        return Identity
    else:
        raise ValueError(f'Unknown mapper name: {name}')
