import faiss
import numpy as np
import torch
from torch import Tensor

from ml.layers.clustering.kmeans import ensure_numpy, KMeans


class KMeans1D:
    centroids: np.ndarray

    def __init__(
            self,
            repr_dim: int, k: int,
            sim: str = 'dotp',
            niter: int = 10, nredo: int = 5,
            gpu: bool = False, verbose: bool = False,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim

        self.kmeans = KMeans(
            repr_dim=1, k=k, sim=sim,
            niter=niter, nredo=nredo,
            gpu=gpu, verbose=verbose
        )

    def fit(self, x: Tensor, weights: Tensor = None, init_centroids: Tensor = None):
        # Convert torch tensors to numpy arrays
        x = ensure_numpy(x)

        # Calculate PCA eigen vector
        self.mat = faiss.PCAMatrix(self.repr_dim, 1)
        self.mat.train(x)
        assert self.mat.is_trained

        # Transform inputs
        tr = self.mat.apply(x)
        if init_centroids is not None:
            init_centroids = self.mat.apply(ensure_numpy(init_centroids))

        self.kmeans.fit(tr, weights, init_centroids)
        self.centroids = self.mat.reverse_transform(self.kmeans.centroids)

    def assign(self, x):
        assert self.centroids is not None, "should train before assigning"
        x = ensure_numpy(x)
        tr = self.mat.apply(x)
        return self.kmeans.assign(tr)

    def get_centroids(self):
        return torch.tensor(self.centroids, dtype=torch.float)