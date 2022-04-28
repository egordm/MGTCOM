from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, List

import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from ml.algo.dpm.priors import NIWPrior, DirichletPrior, Float
from ml.algo.dpm.statistics import DPMMObs, merge_params
from ml.utils import Metric
from shared import get_logger

logger = get_logger(Path(__file__).stem)


SplitDecisions = Union[Tensor, List[bool]]
MergeDecisions = Union[Tensor, List[Tuple[int, int]]]


@dataclass
class MHMC:
    pi_prior: DirichletPrior
    mu_cov_prior: NIWPrior

    ds_scale: float = 1.0
    min_split_points: int = 6
    n_merge_neighbors: int = 3
    metric: Metric = Metric.L2

    def compute_log_h_split(self, obs_c: DPMMObs, obs_sc: DPMMObs) -> Tuple[Float, int]:
        """
        > Compute the log of the hastings probability of splitting a cluster into two clusters, given the data

        :param obs_c: DPMMObs
        :type obs_c: DPMMObs
        :param obs_sc: the observations for the split cluster
        :type obs_sc: DPMMObs
        :return: The log of the probability of the split, and the index of the cluster that is being split.
        """
        lgamma_N_c = torch.lgamma(obs_c.Ns * self.ds_scale)[0]
        lgamma_N_cs = torch.lgamma(obs_sc.Ns * self.ds_scale)

        log_ll_c = self.mu_cov_prior.marginal_log_prob(obs_c.Ns, obs_c.mus, obs_c.covs)[0]
        log_ll_cs = self.mu_cov_prior.marginal_log_prob(obs_sc.Ns, obs_sc.mus, obs_sc.covs)

        H = (
                torch.log(self.pi_prior.params.alpha) + sum(lgamma_N_cs) + sum(log_ll_cs)
                - (lgamma_N_c + log_ll_c)
        )

        return H, log_ll_cs.argmax()

    def check_split(self, obs_c: DPMMObs, obs_sc: DPMMObs) -> Tuple[Float, bool]:
        if sum(obs_c.Ns) < self.min_split_points + 1:
            return -torch.inf, False  # Supercluster is too small

        if (obs_sc.Ns < self.min_split_points).any():
            return -torch.inf, False  # Subclusters are too small

        log_H, _ = self.compute_log_h_split(obs_c, obs_sc)

        # Accept split if H > 1 or with probability H
        return log_H, (log_H > 0 or bool(torch.exp(log_H) > torch.rand(1, device=log_H.device)))

    def check_merge(self, obs_c: DPMMObs, obs_sc: DPMMObs) -> Tuple[Float, bool, int]:
        if (obs_sc.Ns < 1).any():  # One of the subclusters is empty. Always merge.
            return torch.inf, True, obs_sc.Ns.argmax()

        # Compute combined cluster center
        log_H, max_k = self.compute_log_h_split(obs_c, obs_sc)
        log_H = -log_H

        # Accept merge if H > 1 or with probability H
        return log_H, (log_H > 0 or bool(torch.exp(log_H) > torch.rand(1, device=log_H.device))), max_k

    def propose_splits(self, obs_c: DPMMObs, obs_sc: DPMMObs) -> Tuple[SplitDecisions, Tensor]:
        """
        > For each cluster, we check if the cluster should be split by comparing the likelihood of the cluster under the
        current model to the likelihood of the cluster under the proposed model

        :param obs_c: Current observations for the cluster
        :type obs_c: DPMMObs
        :param obs_sc: The observations from the subclusters
        :type obs_sc: DPMMObs
        :return: A boolean tensor of size k, where k is the number of clusters in the current model.
        """
        k = len(obs_c.mus)
        decisions = torch.zeros(k, dtype=torch.bool)
        Hs = torch.zeros(k, dtype=torch.float)

        for i in range(k):
            Hs[i], decisions[i] = self.check_split(
                DPMMObs(obs_c.Ns[[i]], obs_c.mus[[i]], obs_c.covs[[i]]),
                DPMMObs(obs_sc.Ns[i * 2:(i + 1) * 2], obs_sc.mus[i * 2:(i + 1) * 2], obs_sc.covs[i * 2:(i + 1) * 2])
            )

        return decisions, Hs

    def propose_merges(self, obs_c: DPMMObs) -> Tuple[MergeDecisions, Tensor]:
        """
        Compute probable merge candidates by merging k neighboring superclusters.
        :param obs_c:
        :return:
        """
        k = len(obs_c.mus)

        mus = obs_c.mus.cpu()
        neigh = NearestNeighbors(n_neighbors=min(self.n_merge_neighbors, k), metric=self.metric.sk_metric())
        neigh.fit(mus)

        candidates = []
        Ds, Js = neigh.kneighbors(mus, return_distance=True)
        for i, (D, J) in enumerate(zip(Ds, Js)):
            candidates.extend([(d, i, j) for (d, j) in zip(D, J) if j < i])
        candidates = list(sorted(candidates, key=lambda x: x[0]))

        considered = set()
        decisions = []
        Hs = []
        for (_, i, j) in candidates:
            if i in considered or j in considered:
                continue  # Skip if any of the clusters is already considered

            obs_sc = DPMMObs(obs_c.Ns[[i, j]], obs_c.mus[[i, j], :], obs_c.covs[[i, j], :, :])
            obs_c_new = merge_params(obs_sc.Ns, obs_sc.mus, obs_sc.covs)
            H, decision, max_k = self.check_merge(obs_c_new, obs_sc)

            if decision:
                considered.update([i, j])
                decisions.append((i, j) if max_k == 0 else (j, i))
                Hs.append(H)

        return torch.tensor(decisions, dtype=torch.long), torch.tensor(Hs, dtype=torch.float)

    def to(self, device):
        self.mu_cov_prior.to(device)
