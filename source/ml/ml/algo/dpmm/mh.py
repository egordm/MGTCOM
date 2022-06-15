from dataclasses import dataclass
from typing import Tuple, Union, List

import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, lgamma

from ml.algo.dpmm.prior import NWPrior, DirPrior
from ml.algo.dpmm.statistics import GaussianParams, merge_params
from ml.utils import HParams, Metric

SplitDecisions = Union[Tensor, List[bool]]
MergeDecisions = Union[Tensor, List[Tuple[int, int]]]


@dataclass
class MHParams(HParams):
    ds_scale: float = 1.0
    min_split_points: int = 6
    n_merge_neighbors: int = 3
    metric: Metric = Metric.DOTP


@dataclass
class MetropolisHastings:
    hparams: MHParams
    prior_dir: DirPrior
    prior_nw: NWPrior

    def compute_log_h_split(self, params_c: GaussianParams, params_sc: GaussianParams) -> Tuple[Tensor, int]:
        lgamma_N_c = lgamma(params_c.Ns * self.hparams.ds_scale)[0]
        lgamma_N_sc = lgamma(params_sc.Ns * self.hparams.ds_scale)

        log_ll_c = self.prior_nw.estimate_marginal_log_prob(params_c.Ns, params_c.mus, params_c.covs)[0]
        log_ll_sc = self.prior_nw.estimate_marginal_log_prob(params_sc.Ns, params_sc.mus, params_sc.covs)

        H = (
            (self.prior_dir.alpha.log() + sum(lgamma_N_sc) + sum(log_ll_sc))
            - (lgamma_N_c + log_ll_c)
        )

        return H, log_ll_sc.argmax()

    def check_split(self, params_c: GaussianParams, params_sc: GaussianParams) -> Tuple[Tensor, bool]:
        if sum(params_c.Ns) < self.hparams.min_split_points + 1:
            return torch.tensor(-torch.inf), False  # Supercluster is too small

        if (params_sc.Ns < self.hparams.min_split_points).any():
            return torch.tensor(-torch.inf), False  # Subclusters are too small

        log_H, _ = self.compute_log_h_split(params_c, params_sc)

        # Accept split if H > 1 or with probability H
        return log_H, (log_H > 0 or bool(torch.exp(log_H) > torch.rand(1, device=log_H.device)))

    def check_merge(self, params_c: GaussianParams, params_sc: GaussianParams) -> Tuple[Tensor, bool, int]:
        if (params_sc.Ns < 1).any():  # One of the subclusters is empty. Always merge.
            return torch.tensor(torch.inf), True, params_sc.Ns.argmax()

        # Compute combined cluster center
        log_H, max_k = self.compute_log_h_split(params_c, params_sc)
        log_H = -log_H

        # Accept merge if H > 1 or with probability H
        return log_H, (log_H > 0 or bool(torch.exp(log_H) > torch.rand(1, device=log_H.device))), max_k

    def propose_splits(
        self, params_cs: GaussianParams, params_scs: List[GaussianParams]
    ) -> Tuple[SplitDecisions, Tensor]:
        k = len(params_cs.mus)
        decisions = torch.zeros(k, dtype=torch.bool)
        Hs = torch.zeros(k, dtype=torch.float)

        for i in range(k):
            Hs[i], decisions[i] = self.check_split(
                GaussianParams(params_cs.Ns[[i]], params_cs.mus[[i]], params_cs.covs[[i]]),
                params_scs[i],
            )

        return decisions, Hs

    def propose_merges(self, params_cs: GaussianParams) -> Tuple[MergeDecisions, Tensor]:
        k = len(params_cs.mus)

        mus = params_cs.mus.cpu()
        neigh = NearestNeighbors(
            n_neighbors=min(self.hparams.n_merge_neighbors, k),
            metric=self.hparams.metric.sk_metric()
        )
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

            obs_sc = GaussianParams(params_cs.Ns[[i, j]], params_cs.mus[[i, j], :], params_cs.covs[[i, j], :, :])
            obs_c_new = merge_params(obs_sc.Ns, obs_sc.mus, obs_sc.covs)
            H, decision, max_k = self.check_merge(obs_c_new, obs_sc)

            if decision:
                considered.update([i, j])
                decisions.append((i, j) if max_k == 0 else (j, i))
                Hs.append(H)

        return torch.tensor(decisions, dtype=torch.long), torch.tensor(Hs, dtype=torch.float)
