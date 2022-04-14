import copy
from enum import Enum
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor


class WeightsInitMode(Enum):
    Same = 'same'
    Random = 'random'


class ClusteringNet(torch.nn.Module):
    def __init__(self, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.k = k

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_net = torch.nn.Linear(self.in_dim, self.out_dim)
        self.out_net = torch.nn.Linear(self.out_dim, self.k)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.in_net(x))
        x = self.out_net(x)
        r = F.softmax(x, dim=1)
        return r

    def split(self, decisions: Tensor, split_mode: WeightsInitMode):
        out_net = self.out_net
        split_idx = torch.nonzero(decisions, as_tuple=False)
        k_new = self.k + len(split_idx)

        with torch.no_grad():
            self.out_net = torch.nn.Linear(self.out_dim, k_new)

            weights_not_split, weights_split = out_net.weight.data[~decisions, :], out_net.weight.data[decisions, :]
            bias_not_split, bias_split = out_net.bias.data[~decisions], out_net.bias.data[decisions]

            weights_new, bias_new = split_weights(weights_split, bias_split, split_mode)

            self.out_net.weight.data = torch.cat([weights_not_split, weights_new], dim=0)
            self.out_net.bias.data = torch.cat([bias_not_split, bias_new], dim=0)
            self.k = k_new

            del out_net

    def merge(self, decisions: Tensor, merge_mode: WeightsInitMode):
        out_net = self.out_net
        k_new = self.k - len(decisions)

        mask = torch.zeros(self.k, dtype=torch.bool, device=out_net.weight.device)
        mask[decisions.flatten()] = True
        pref_idx = decisions[:, 0]

        with torch.no_grad():
            self.out_net = torch.nn.Linear(self.out_dim, k_new)

            weights_not_merge, weights_merge = out_net.weight.data[~mask, :], out_net.weight.data[pref_idx, :]
            bias_not_merge, bias_merge = out_net.bias.data[~mask], out_net.bias.data[pref_idx]

            weights_new, bias_new = merge_weights(weights_merge, bias_merge, merge_mode)

            self.out_net.weight.data = torch.cat([weights_not_merge, weights_new], dim=0)
            self.out_net.bias.data = torch.cat([bias_not_merge, bias_new], dim=0)
            self.k = k_new

            del out_net


class SubClusteringNet(torch.nn.Module):
    def __init__(self, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.k = k

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.out_dim, 2),
            )
            for _ in range(self.k)
        ])

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        ri = torch.full((len(x), 2), float('-inf'), dtype=torch.float, device=x.device)
        for i in range(self.k):
            if sum(z == i) > 0:
                ri[z == i, :] = self.net[i](x[z == i])

        return F.softmax(ri, dim=-1)

    def split(self, decisions: Tensor, split_mode: WeightsInitMode):
        net = self.net
        split_idx = torch.nonzero(decisions, as_tuple=False)
        k_new = self.k + len(split_idx)

        with torch.no_grad():
            new_net = []
            for i in (~decisions).nonzero():
                new_net.append(net[int(i)])
            for i in (decisions).nonzero():
                if split_mode == WeightsInitMode.Same:
                    for _ in range(2):
                        new_net.append(copy.deepcopy(net[int(i)]))
                elif split_mode == WeightsInitMode.Random:
                    for _ in range(2):
                        new_net.append(torch.nn.Sequential(
                            torch.nn.Linear(self.in_dim, self.out_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.out_dim, 2),
                        ))

            self.net = torch.nn.ModuleList(new_net)
            self.k = k_new
            del net

    def merge(self, decisions: Tensor, merge_mode: WeightsInitMode):
        net = self.net
        k_new = self.k - len(decisions)

        mask = torch.zeros(self.k, dtype=torch.bool)
        mask[decisions.flatten()] = True
        pref_idx = decisions[:, 0]

        with torch.no_grad():
            new_net = []
            for i in (~mask).nonzero():
                new_net.append(net[int(i)])
            for i in pref_idx:
                if merge_mode == WeightsInitMode.Same:
                    new_net.append(net[int(i)])
                elif merge_mode == WeightsInitMode.Random:
                    new_net.append(torch.nn.Sequential(
                        torch.nn.Linear(self.in_dim, self.out_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.out_dim, 2),
                    ))

            self.net = torch.nn.ModuleList(new_net)
            self.k = k_new
            del net


def split_weights(weight: Tensor, bias: Tensor, split_mode: WeightsInitMode) -> Tuple[Tensor, Tensor]:
    if split_mode == WeightsInitMode.Same:
        return (
            weight.repeat_interleave(2, dim=0),
            bias.repeat_interleave(2, dim=0),
        )
    elif split_mode == WeightsInitMode.Random:
        return (
            torch.nn.init.xavier_uniform_(torch.FloatTensor(len(weight) * 2, *weight.shape[1:])),
            torch.zeros(len(bias) * 2, *bias.shape[1:]),
        )
    else:
        raise NotImplementedError


def merge_weights(weight: Tensor, bias: Tensor, merge_mode: WeightsInitMode) -> Tuple[Tensor, Tensor]:
    if merge_mode == WeightsInitMode.Same:
        return (
            weight,
            bias,
        )
    elif merge_mode == WeightsInitMode.Random:
        return (
            torch.nn.init.xavier_uniform_(torch.empty_like(weight)),
            torch.zeros_like(bias),
        )
    else:
        raise NotImplementedError
