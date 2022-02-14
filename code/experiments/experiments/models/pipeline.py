from typing import Tuple

import torch
import torchmetrics
from torch_geometric.data import HeteroData

import ml
from experiments.losses.clustering import ClusterCohesionLoss, NegativeEntropyRegularizer
from experiments.models.clustering import ClusteringModule
from experiments.models.embedding import LinkPredictionModule, EmbeddingModule


