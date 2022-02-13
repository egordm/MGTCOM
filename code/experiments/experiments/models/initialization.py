import faiss
import pytorch_lightning as pl
import torch

from experiments.models.embedding import EmbeddingModule
from experiments.models.pipeline import EmbeddingNet


class KMeansInitializer:
    def __init__(self, repr_dim: int = 32, k: int = 5, dist = 'cosine', **kwargs) -> None:
        super().__init__()
        self.kmeans = faiss.Kmeans(repr_dim, k, **kwargs)
        self.dist = dist

    def fit(self, trainer: pl.Trainer, embedding_module: EmbeddingModule, data_module: pl.LightningDataModule):
        model = EmbeddingNet(embedding_module)
        predictions = trainer.predict(model, data_module)
        embeddings = torch.cat(predictions, dim=0).detach().cpu()
        if self.dist == 'cosine':
            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings.numpy()
        self.kmeans.train(embeddings)

        assignment = self.kmeans.index.assign(embeddings, 1)
        return assignment.squeeze(), self.kmeans.centroids
