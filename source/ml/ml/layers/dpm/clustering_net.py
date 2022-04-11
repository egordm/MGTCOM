import torch
import torch.nn.functional as F
from torch import Tensor


class ClusteringNet(torch.nn.Module):
    def __init__(self, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.k = k

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_net = torch.nn.Linear(in_dim, self.out_dim)
        self.out_net = torch.nn.Linear(self.out_dim, k)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.in_net(x))
        x = self.out_net(x)
        r = F.softmax(x, dim=1)
        return r


class SubClusteringNet(torch.nn.Module):
    def __init__(self, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.k = k

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_net = torch.nn.Linear(in_dim, self.out_dim * k)
        self.out_net = torch.nn.Linear(self.out_dim * k, k * 2)

        # Detach different clusters by zeroing out their interactions
        out_gradient_mask = torch.zeros(self.out_dim * k, k * 2)
        for i in range(k):
            out_gradient_mask[self.out_dim * i:self.out_dim * (i + 1), i * 2:(i + 1) * 2] = 1

        self.out_net.weight.data *= out_gradient_mask.T
        self.out_net.weight.register_hook(
            lambda grad: grad.mul_(out_gradient_mask.T.to(device=grad.device))
        )

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = F.relu(self.in_net(x))
        ri = self.out_net(x)

        # Zero our irrelevant (non z) clusters
        mask = torch.zeros_like(ri)
        mask[torch.arange(len(z)), 2 * z] = 1.
        mask[torch.arange(len(z)), 2 * z + 1] = 1.

        # Masked softmax
        ri = F.softmax(ri.masked_fill((1 - mask).bool(), float('-inf')), dim=1)
        return ri
