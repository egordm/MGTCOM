import faiss
import torch
from torch import Tensor


class KMeans:
    clus: faiss.Clustering
    index: faiss.Index

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
        elif sim == 'l2':
            self.index_cls = faiss.IndexFlatL2
        elif sim == 'cos':
            self.params.spherical = True
            self.index_cls = faiss.IndexFlatIP
        else:
            raise ValueError(f"Unknown similarity {sim}")

    def fit(self, x: Tensor, weights: Tensor = None, init_centroids: Tensor = None):
        # Convert torch tensors to numpy arrays
        x = x.cpu().numpy()
        if weights is not None:
            weights = weights.cpu().numpy()
        if init_centroids is not None:
            init_centroids = init_centroids.cpu().numpy()

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

    def assign(self, x):
        assert self.centroids is not None, "should train before assigning"
        x = x.cpu().numpy()
        self.index.reset()
        self.index.add(self.centroids)
        return torch.tensor(self.index.assign(x, k=1).ravel(), dtype=torch.long)
