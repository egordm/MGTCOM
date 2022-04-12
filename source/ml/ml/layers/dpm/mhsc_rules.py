from typing import Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.neighbors import NearestNeighbors

from ml.layers.dpm import Priors, StackedGaussianMixtureModel, GaussianMixtureModel


class MHSCRules:
    """
    Metropolis-Hastings Sub-Cluster Split and Merge Moves

    See Section 5.3:
        J. Chang and J. W. Fisher III, “Parallel Sampling of DP Mixture Models using Sub-Cluster Splits,”
        in Advances in Neural Information Processing Systems, 2013, vol. 26. Accessed: Apr. 09, 2022. [Online].
        Available: https://papers.nips.cc/paper/2013/hash/bca82e41ee7b0833588399b1fcd177c7-Abstract.html

    """

    def __init__(
            self,
            prior: Priors, gmm: GaussianMixtureModel, gmm_sub: StackedGaussianMixtureModel,
            alpha: float, split_prob: Optional[float], merge_prob: Optional[float],
            min_split_points: int = 6, n_merge_neighbors=3, sim='euclidean',
    ) -> None:
        super().__init__()
        self.prior = prior
        self.gmm = gmm
        self.gmm_sub = gmm_sub
        self.alpha = alpha
        self.split_prob = split_prob
        self.merge_prob = merge_prob
        self.min_split_points = min_split_points
        self.sim = sim
        self.n_merge_neighbors = n_merge_neighbors

    def compute_log_h_split(self, X: Tensor, X_K: Tensor, N_K: Tensor, mu: Tensor, mu_sub: Tensor):
        # See Eq (21)
        N_c = sum(N_K)

        log_ll_c = self.prior.log_marginal_likelihood(X, mu)
        log_ll_K = torch.tensor([self.prior.log_marginal_likelihood(X_k, mu_sub[k, :]) for k, X_k in enumerate(X_K)])

        lgamma_N_K = [torch.lgamma(N_k) if N_k > 0 else 0 for N_k in N_K]
        lgamma_N_c = torch.lgamma(N_c)

        # Note: np.log(self.alpha) is changed to nonlog. I have a few small datasets
        H = (
                (self.alpha + sum(lgamma_N_K) + sum(log_ll_K))
                - (lgamma_N_c + log_ll_c)
        )

        print(H)
        return H, log_ll_K.argmax(dim=-1)

    def split_rule(self, X: Tensor, ri: Tensor, mu: Tensor, mu_sub: Tensor):
        # Too few points
        if len(X) < self.min_split_points + 1:
            return False

        k = len(mu_sub)
        I = ri.argmax(dim=-1)
        X_K = [X[I == i] for i in range(k)]
        N_K = torch.tensor([len(X_k) for X_k in X_K], dtype=torch.float)

        # Subclusters are too small
        if any(N_k < self.min_split_points for N_k in N_K):
            return False

        # Compute Hastings ratio
        H, _ = self.compute_log_h_split(X, X_K, N_K, mu, mu_sub)

        # Accept split if H > 0 or with probability split_prob
        split_prob = self.split_prob or torch.exp(H)
        return bool(H > 0 or split_prob > torch.rand(1))

    def merge_rule(self, X: Tensor, ri: Tensor, mus: Tensor):
        k = len(mus)
        I = ri.argmax(dim=-1)
        X_K = [X[I == i] for i in range(k)]
        N_K = torch.tensor([len(X_k) for X_k in X_K], dtype=torch.float)
        N_c = sum(N_K)

        # Compute combined cluster center
        if N_c > 0:
            mu = (N_K.reshape(-1, 1) * mus).sum(dim=0) / N_c  # Weighted mean
        else:
            mu = mus.mean(dim=0)

        # H_merge = 1 / H_merge
        H, max_k = self.compute_log_h_split(X, X_K, N_K, mu, mus)
        H = -H

        # Accept merge if H > 0 or with probability merge_prob
        merge_prob = self.merge_prob or torch.exp(H)
        return bool(H > 0 or merge_prob > torch.rand(1)), max_k

    def split_decisions(self, X: Tensor, r: Tensor, ri: Tensor) -> Tensor:
        decisions = torch.zeros(self.gmm.n_components, dtype=torch.bool)

        z = r.argmax(dim=-1)
        for i in range(self.gmm.n_components):
            X_k = X[z == i]
            ri_k = ri[z == i, self.gmm_sub.n_subcomponents * i: self.gmm_sub.n_subcomponents * (i + 1)]

            decisions[i] = self.split_rule(X_k, ri_k, self.gmm.mus.data[i], self.gmm_sub.component(i).mus.data)

        return decisions

    def merge_decisions(self, X: Tensor, r: Tensor) -> Tensor:
        # Compute probable merge candidates by merging the most neighboring subclusters
        neigh = NearestNeighbors(n_neighbors=min(self.n_merge_neighbors, self.gmm.n_components), metric=self.sim)
        neigh.fit(self.gmm.mus.data)

        candidates = []
        ds, js = neigh.kneighbors(self.gmm.mus.data, return_distance=True)
        for i in range(self.gmm.n_components):
            candidates.extend([(d, i, j) for (d, j) in zip(ds[i], js[i]) if j < i])
        candidates = list(sorted(candidates, key=lambda x: x[0]))

        # Compute merge decisions
        z = r.argmax(dim=-1)
        considered = set()
        decisions = []
        for (_, i, j) in candidates:
            # Skip if any of the clusters is already considered
            if i in considered or j in considered:
                continue

            X_K = X[torch.logical_or(z == i, z == j)]
            r_K = r[:, [i, j]]
            mus = self.gmm.mus.data[[i, j], :]

            decision, dominant_k = self.merge_rule(X_K, r_K, mus)
            if decision:
                considered.add(i)
                considered.add(j)
                decisions.append((i, j) if dominant_k == 0 else (j, i))

        return decisions
