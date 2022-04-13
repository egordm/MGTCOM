import faiss
import numpy as np
import torch
from torch import Tensor

from ml.utils.tensor import ensure_numpy


class KMeans:
    clus: faiss.Clustering
    index: faiss.Index
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
        self.k = k
        self.gpu = gpu
        self.verbose = verbose

        self.params = faiss.ClusteringParameters()
        self.params.niter = niter
        self.params.nredo = nredo
        self.params.verbose = verbose

        if sim == 'dotp':
            self.index_cls = faiss.IndexFlatIP
        elif sim == 'euclidean':
            self.index_cls = faiss.IndexFlatL2
        elif sim == 'cos':
            self.params.spherical = True
            self.index_cls = faiss.IndexFlatIP
        else:
            raise ValueError(f"Unknown similarity {sim}")

    def fit(self, x: Tensor, weights: Tensor = None, init_centroids: Tensor = None):
        # Convert torch tensors to numpy arrays
        x = ensure_numpy(x)
        if weights is not None:
            weights = ensure_numpy(weights)
        if init_centroids is not None:
            init_centroids = ensure_numpy(init_centroids)

        # Initialize clustering
        self.clus = faiss.Clustering(self.repr_dim, self.k, self.params)
        if init_centroids is not None:
            nc, d2 = init_centroids.shape
            assert d2 == self.repr_dim
            faiss.copy_array_to_vector(init_centroids.ravel(), self.clus.centroids)

        # Initialize index
        self.index = self.index_cls(self.repr_dim)
        if self.gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index, ngpu=self.gpu)

        # Train KMeans
        self.clus.train(x, self.index, weights)
        centroids = faiss.vector_float_to_array(self.clus.centroids)
        self.centroids = centroids.reshape(self.k, self.repr_dim)

        return self

    def assign(self, x):
        assert self.centroids is not None, "should train before assigning"
        x = ensure_numpy(x)
        self.index.reset()
        self.index.add(self.centroids)
        return torch.tensor(self.index.assign(x, k=1).ravel(), dtype=torch.long)

    def get_centroids(self):
        return torch.tensor(self.centroids, dtype=torch.float)
