from enum import Enum
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor


class SplitMode(Enum):
    Same = 'same'
    Random = 'random'


class MergeMode(Enum):
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

    def split(self, decisions: Tensor, split_mode: SplitMode):
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

    def merge(self, decisions: Tensor, merge_mode: MergeMode):
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


class SubClusteringNet(torch.nn.Module):
    def __init__(self, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.k = k

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_net = torch.nn.Linear(self.in_dim, self.out_dim * self.k)
        self.out_net = torch.nn.Linear(self.out_dim * self.k, self.k * 2)
        self._decouple_net()

    def _decouple_net(self):
        # Detach different clusters by zeroing out their interactions
        out_gradient_mask = torch.zeros(self.out_dim * self.k, self.k * 2)
        for i in range(self.k):
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

    def split(self, decisions: Tensor, split_mode: SplitMode):
        in_net = self.in_net
        out_net = self.out_net
        split_idx = torch.nonzero(decisions, as_tuple=False)
        k_new = self.k + len(split_idx)

        with torch.no_grad():
            self.in_net = torch.nn.Linear(self.in_dim, self.out_dim * k_new)
            in_weights = in_net.weight.data.view(self.k, self.out_dim, self.in_dim)
            in_weights_not_split, in_weights_split = in_weights[~decisions, :, :], in_weights[decisions, :, :]
            in_bias = in_net.bias.data.view(self.k, self.out_dim)
            in_bias_not_split, in_bias_split = in_bias[~decisions], in_bias[decisions]

            in_weights_new, in_bias_new = split_weights(in_weights_split, in_bias_split, split_mode)

            self.in_net.weight.data = torch.cat([in_weights_not_split, in_weights_new], dim=0).view(k_new * self.out_dim, self.in_dim)
            self.in_net.bias.data = torch.cat([in_bias_not_split, in_bias_new], dim=0).view(k_new * self.out_dim)

            self.out_net = torch.nn.Linear(self.out_dim * k_new, k_new * 2)
            out_weights = out_net.weight.data.view(self.k, 2, self.k, self.out_dim)
            out_weights_not_split, out_weights_split = (
                out_weights[~decisions, :, :, :][:, :, ~decisions, :],
                out_weights[decisions, :, :, :][:, :, decisions, :],
            )
            out_bias = out_net.bias.data.view(self.k, 2)
            out_bias_not_split, out_bias_split = out_bias[~decisions, :], out_bias[decisions, :, ]

            out_weights_new, out_bias_new = split_weights(out_weights_split, out_bias_split, split_mode)

            self.out_net.weight.data.view(k_new, 2, k_new, self.out_dim)[:out_weights_not_split.size(0), :, :out_weights_not_split.size(2), :] = out_weights_not_split
            self.out_net.weight.data.view(k_new, 2, k_new, self.out_dim)[out_weights_not_split.size(0):, :, out_weights_not_split.size(2):, :] = out_weights_new
            self.out_net.bias.data.view(k_new, 2)[:, :] = torch.cat([out_bias_not_split, out_bias_new], dim=0)

            self.k = k_new
            self._decouple_net()

    def merge(self, decisions: Tensor, merge_mode: MergeMode):
        in_net = self.in_net
        out_net = self.out_net
        k_new = self.k - len(decisions)

        mask = torch.zeros(self.k, dtype=torch.bool, device=out_net.weight.device)
        mask[decisions.flatten()] = True
        pref_idx = decisions[:, 0]

        with torch.no_grad():
            self.in_net = torch.nn.Linear(self.in_dim, self.out_dim * k_new)
            in_weights = in_net.weight.data.view(self.k, self.out_dim, self.in_dim)
            in_weights_not_merge, in_weights_merge = in_weights[~mask, :, :], in_weights[pref_idx, :, :]
            in_bias = in_net.bias.data.view(self.k, self.out_dim)
            in_bias_not_merge, in_bias_merge = in_bias[~mask], in_bias[pref_idx, :]

            in_weights_new, in_bias_new = merge_weights(in_weights_merge, in_bias_merge, merge_mode)

            self.in_net.weight.data.view(k_new, self.out_dim, self.in_dim)[:, :, :] = torch.cat([in_weights_not_merge, in_weights_new], dim=0)
            self.in_net.bias.data.view(k_new, self.out_dim)[:, :] = torch.cat([in_bias_not_merge, in_bias_new], dim=0)

            self.out_net = torch.nn.Linear(self.out_dim * k_new, k_new * 2)
            out_weights = out_net.weight.data.view(self.k, 2, self.k, self.out_dim)
            out_weights_not_merge, out_weights_merge = (
                out_weights[~mask, :, :, :][:, :, ~mask, :],
                out_weights[pref_idx, :, :, :][:, :, pref_idx, :],
            )
            out_bias = out_net.bias.data.view(self.k, 2)
            out_bias_not_merge, out_bias_merge = out_bias[~mask, :], out_bias[pref_idx, :]

            out_weights_new, out_bias_new = merge_weights(out_weights_merge, out_bias_merge, merge_mode)

            self.out_net.weight.data.view(k_new, 2, k_new, self.out_dim)[:out_weights_not_merge.size(0), :, :out_weights_not_merge.size(2), :] = out_weights_not_merge
            self.out_net.weight.data.view(k_new, 2, k_new, self.out_dim)[out_weights_not_merge.size(0):, :, out_weights_not_merge.size(2):, :] = out_weights_new
            self.out_net.bias.data.view(k_new, 2)[:, :] = torch.cat([out_bias_not_merge, out_bias_new], dim=0)

            self.k = k_new
            self._decouple_net()



def split_weights(weight: Tensor, bias: Tensor, split_mode: SplitMode) -> Tuple[Tensor, Tensor]:
    if split_mode == SplitMode.Same:
        return (
            weight.repeat_interleave(2, dim=0),
            bias.repeat_interleave(2, dim=0),
        )
    elif split_mode == SplitMode.Random:
        return (
            torch.FloatTensor(len(weight) * 2, *weight.shape[1:]).uniform_(-1, 1),
            torch.FloatTensor(len(bias) * 2, *bias.shape[1:]),
        )
    else:
        raise NotImplementedError


def merge_weights(weight: Tensor, bias: Tensor, merge_mode: MergeMode) -> Tuple[Tensor, Tensor]:
    if merge_mode == MergeMode.Same:
        return (
            weight,
            bias,
        )
    elif merge_mode == MergeMode.Random:
        return (
            torch.empty_like(weight).uniform_(-1, 1),
            torch.empty_like(bias),
        )
    else:
        raise NotImplementedError
