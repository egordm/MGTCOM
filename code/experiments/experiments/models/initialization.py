import faiss
import pytorch_lightning as pl
import torch

from experiments.models.embedding import EmbeddingModule


class KMeansInitializer:
    def __init__(self, repr_dim: int = 32, k: int = 5, dist = 'cosine', **kwargs) -> None:
        super().__init__()
        self.kmeans = faiss.Kmeans(repr_dim, k, **kwargs)
        self.dist = dist

    def fit(self, embeddings: torch.Tensor) -> None:
        if self.dist == 'cosine':
            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings.numpy()
        self.kmeans.train(embeddings)

        assignment = self.kmeans.index.assign(embeddings, 1)
        return assignment.squeeze(), self.kmeans.centroids
