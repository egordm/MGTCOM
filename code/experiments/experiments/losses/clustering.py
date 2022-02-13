import math
from typing import List

import torch


class ClusterCohesionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    def forward(self, q_l: torch.Tensor, q_r: torch.Tensor, label: torch.Tensor):
        sim = (self.cos_sim(q_l, q_r) + 1) / 2
        pred = torch.stack([1 - sim, sim], dim=1)
        loss = self.ce_loss(pred, label)
        return loss


class NegativeEntropyRegularizer(torch.nn.Module):
    def forward(self, *args: List[torch.Tensor]):
        p = sum(map(lambda q: torch.sum(q, dim=0), args))
        p /= p.sum()
        ne = math.log(p.size(0)) + (p * torch.log(p)).sum()
        return ne
