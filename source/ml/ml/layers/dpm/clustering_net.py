import torch
import torch.nn.functional as F


class ClusteringNet(torch.nn.Module):
    def __init__(self, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.k = k

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_net = torch.nn.Linear(in_dim, self.out_dim)
        self.out_net = torch.nn.Linear(self.out_dim, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in_net(x))
        x = self.out_net(x)
        ri = F.softmax(x, dim=1)
        return ri


class SubClusteringNet(torch.nn.Module):
    def __init__(self, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.k = k

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_net = torch.nn.Linear(in_dim, self.out_dim * k)
        self.out_net = torch.nn.Linear(self.out_dim * k, k * 2)

        out_gradient_mask = torch.zeros(self.out_dim * k, k * 2)
        for i in range(k):
            out_gradient_mask[self.out_dim * i:self.out_dim * (i + 1), i * 2:(i + 1) * 2] = 1
        self.out_net.weight.data = out_gradient_mask

        self.out_net.weight.register_hook(
            lambda grad: grad.mul_(out_gradient_mask.T.to(device=self.device))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in_net(x))
        ris = self.out_net(x)
        return ris
