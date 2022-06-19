from copy import copy
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple, List, Tuple, Any

import torch
from torch import Tensor

from ml.algo.dpmm.base import BaseMixture
from ml.algo.dpmm.dpm import DPMMParams, DirichletProcessMixture, DirichletProcessMixtureParams
from ml.algo.dpmm.mh import MetropolisHastings, MHParams
from ml.algo.dpmm.prior import DirPrior, NWPrior
from ml.algo.dpmm.statistics import InitMode, estimate_gaussian_parameters, GaussianParams
from ml.models.base.base_model import BaseModel
from ml.utils import unique_count, mask_from_idx
from ml.utils.training import ClusteringStage
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class Action(IntEnum):
    Split = 0
    Merge = 1
    NoAction = 2


class DPMSCParams(NamedTuple):
    cluster: DPMMParams
    subcluster: List[DPMMParams]


@dataclass
class DPMSCHParams(DirichletProcessMixtureParams, MHParams):
    mutate: bool = True
    subcluster: bool = True


# noinspection PyProtectedMember
class DPMSC(BaseMixture[DPMSCParams]):
    hparams: DPMSCHParams

    def __init__(self, hparams: DPMSCHParams) -> None:
        super().__init__(hparams)
        self.max_sub_reinit = 2

        self.clusters = DirichletProcessMixture(hparams)
        self.subclusters = [
            self._create_subcluster()
            for _ in range(self.n_components)
        ]
        self.mh = None
        self.reinit_count = [0 for _ in range(self.n_components)]
        self.prev_action = Action.NoAction

    def _create_subcluster(self) -> DirichletProcessMixture:
        hparams = copy(self.hparams)
        hparams.init_k = 2
        hparams.init_mode = InitMode.KMEANS1D

        ret = DirichletProcessMixture(hparams)
        ret.prior_dir = DirPrior.from_params(1.0 / 2.0)
        if self.clusters.prior_nw is not None:
            ret.prior_nw = self.clusters.prior_nw

        return ret

    def _init(self, X: Tensor) -> None:
        self.clusters._init(X)
        for subcluster in self.subclusters:
            subcluster.prior_dir = DirPrior.from_params(1.0 / 2.0)
            subcluster.prior_nw = self.clusters.prior_nw

        self.mh = MetropolisHastings(self.hparams, self.clusters.prior_dir, self.clusters.prior_nw)

    def _init_params(self, X: Tensor, z_init: Tensor = None) -> None:
        if z_init is not None:
            k = len(torch.unique(z_init))
            if k != self.n_components:
                self.n_components = k
                self.clusters.n_components = k
                self.subclusters = [
                    self._create_subcluster()
                    for _ in range(self.n_components)
                ]
                self.reinit_count = [0 for _ in range(self.n_components)]

        self.clusters._init_params(X, z_init)
        z = self.clusters.predict(X)
        Ns = unique_count(z, self.n_components)
        for i, subcluster in enumerate(self.subclusters):
            subcluster._init_params(X[z == i] if Ns[i] > 2 else X)

    def _e_step(self, X: Tensor) -> Tuple[Tuple[Tensor, List[Tensor]], Tuple[Tensor, List[Tensor]]]:
        log_prob_norm, log_prob = super()._e_step(X)
        z = log_prob.argmax(dim=1)

        log_prob_norm_sub, log_prob_sub = [], []
        for i, subcluster in enumerate(self.subclusters):
            X_i = X[z == i]
            log_prob_norm_i, log_prob_i = subcluster._e_step(X_i)
            log_prob_norm_sub.append(log_prob_norm_i)
            log_prob_sub.append(log_prob_i)

        return (log_prob_norm, log_prob_norm_sub), (log_prob, log_prob_sub)

    def _m_step(self, X: Tensor, log_r: Tuple[Tensor, List[Tensor]]) -> None:
        log_r, log_r_sub = log_r

        if self._mutate_clean_superclusters(X, log_r):
            pass
        else:
            self.clusters._m_step(X, log_r)

            z = log_r.argmax(dim=1)
            for i, subcluster in enumerate(self.subclusters):
                zi = log_r_sub[i].argmax(dim=1)
                Ns_i = unique_count(zi, 2)
                if ((Ns_i / Ns_i.sum()) < 0.1).any() and self.reinit_count[i] < self.max_sub_reinit:
                    logger.warning(f"Encountered a saturated subcluster. Reinitializing.")
                    subcluster._init_params(X[z == i])
                    self.reinit_count[i] += 1
                else:
                    subcluster._m_step(X[z == i], log_r_sub[i])

    def _estimate_log_weights(self) -> Tensor:
        return self.clusters._estimate_log_weights()

    def _estimate_log_prob(self, X: Tensor) -> Tensor:
        return self.clusters._estimate_log_prob(X)

    def _compute_lower_bound(self, X, log_r: Tuple[Tensor, List[Tensor]]) -> Tensor:
        log_r, log_r_sub = log_r
        z = log_r.argmax(dim=1)

        lower_bound = self.clusters._compute_lower_bound(X, log_r)
        for i, subcluster in enumerate(self.subclusters):
            lower_bound += subcluster._compute_lower_bound(X[z == i], log_r_sub[i])

        return lower_bound

    def predict_full(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.predict(X)
        zi = torch.zeros_like(z)
        for i, subcluster in enumerate(self.subclusters):
            zi[z == i] = subcluster.predict(X[z == i])

        return z, zi

    @property
    def cluster_params(self) -> GaussianParams:
        return self.clusters.cluster_params

    @property
    def subcluster_params(self) -> List[GaussianParams]:
        return [cluster.cluster_params for cluster in self.subclusters]

    def _on_converge(self, X: Tensor, log_r: Tensor) -> bool:
        log_r, log_r_sub = log_r
        changed = self._mutate(X, log_r, log_r_sub)
        print(f'Changed during mutation: {changed}')
        return not changed

    def _mutate(self, X: Tensor, log_r: Tensor, log_r_sub: List[Tensor]) -> bool:
        if not self.hparams.mutate:
            return False

        z = log_r.argmax(dim=1)
        r_hard = torch.zeros_like(log_r)
        r_hard[torch.arange(len(r_hard)), z] = 1
        params_cs = estimate_gaussian_parameters(X, r_hard, self.hparams.reg_cov)

        params_scs = []
        for i in range(self.n_components):
            r_sub_hard = torch.zeros_like(log_r_sub[i])
            r_sub_hard[torch.arange(len(r_sub_hard)), log_r_sub[i].argmax(dim=-1)] = 1
            params_scs.append(estimate_gaussian_parameters(X[z == i], r_sub_hard, self.hparams.reg_cov))

        result = False
        actions = [Action.Split, Action.Merge] if self.prev_action != Action.Split else [Action.Merge, Action.Split]
        for action in actions:
            self.prev_action = action
            if action == Action.Split:
                result = self._mutate_split(
                    X, log_r, log_r_sub,
                    params_cs, params_scs
                )
            elif action == Action.Merge:
                result = self._mutate_merge(
                    X, log_r, log_r_sub,
                    params_cs, params_scs
                )

            if result:
                break

        return result

    def _mutate_split(
        self,
        X: Tensor, log_r: Tensor, log_r_sub: List[Tensor],
        params_cs: GaussianParams, params_scs: List[GaussianParams]
    ) -> bool:
        decisions, Hs = self.mh.propose_splits(params_cs, params_scs)
        logger.info("Proposed splits: \n{}".format(
            '\n'.join(map(str, enumerate(zip(decisions.tolist(), Hs.tolist()))))
        ))

        if not decisions.any():
            return False

        # Split superclusters
        new_log_r = [log_r[:, ~decisions]]
        for i in decisions.nonzero().flatten():
            _, log_r_sub_i = self.subclusters[i]._estimate_log_prob_resp(X)
            new_log_r.append(log_r_sub_i + log_r[:, i][:, None])

        new_log_r = torch.cat(new_log_r, dim=1)
        self.clusters._m_step(X, new_log_r)

        # Split subclusters
        z = log_r.argmax(dim=1)
        self.subclusters = [
            self.subclusters[i]
            for i in (~decisions).nonzero().flatten()
        ]
        for i in decisions.nonzero().flatten():
            z_sub = log_r_sub[i].argmax(dim=-1)
            for j in range(2):
                X_sub = X[z == i][z_sub == j]
                subcluster = self._create_subcluster()
                subcluster._init_params(X_sub)
                self.subclusters.append(subcluster)

        # Ensure that params are up to date
        self._set_params(self._get_params())
        self.reinit_count = [0 for _ in range(self.n_components)]

        return True

    def _mutate_merge(
        self,
        X: Tensor, log_r: Tensor, log_r_sub: List[Tensor],
        params_cs: GaussianParams, params_scs: List[GaussianParams]
    ) -> bool:
        if self.n_components < 2:
            return False

        pairs, Hs = self.mh.propose_merges(params_cs)
        logger.info("Proposed merges: \n{}".format(
            '\n'.join(map(str, zip(pairs.tolist(), Hs.tolist())))
        ))

        if len(pairs) == 0:
            return False

        decisions = mask_from_idx(pairs.flatten(), self.n_components)

        # Merge superclusters
        new_log_r = [log_r[:, ~decisions]]
        for pair in pairs:
            new_log_r.append(log_r[:, pair].sum(dim=1, keepdim=True))

        new_log_r = torch.cat(new_log_r, dim=1)
        self.clusters._m_step(X, new_log_r)

        # Merge subclusters
        z = log_r.argmax(dim=1)
        self.subclusters = [
            self.subclusters[i]
            for i in (~decisions).nonzero().flatten()
        ]
        for pair in pairs:
            X_super = X[torch.logical_or(z == pair[0], z == pair[1])]
            subcluster = self._create_subcluster()
            subcluster._init_params(X_super)
            self.subclusters.append(subcluster)

        # Ensure that params are up to date
        self._set_params(self._get_params())
        self.reinit_count = [0 for _ in range(self.n_components)]

        return True

    def _mutate_clean_superclusters(self, X: Tensor, log_r: Tensor) -> bool:
        z = log_r.argmax(dim=1)
        Ns = unique_count(z, self.n_components)

        if (Ns > 1).all():
            return False

        decisions = (Ns <= 1)
        logger.info(f'Removing empty clusters: \n{decisions.nonzero().flatten().tolist()}')

        # Remove superclusters
        new_log_r = log_r[:, ~decisions]
        new_log_r = new_log_r - torch.logsumexp(new_log_r, dim=1)[:, None]

        self.clusters._m_step(X, new_log_r)

        # Remove subclusters
        self.subclusters = [
            self.subclusters[i]
            for i in (~decisions).nonzero().flatten()
        ]
        self.reinit_count = [self.reinit_count[i] for i in (~decisions).nonzero().flatten()]

        # Ensure that params are up to date
        self._set_params(self._get_params())

        return True

    def _get_params(self) -> DPMSCParams:
        return DPMSCParams(
            self.clusters._get_params(),
            [subcluster._get_params() for subcluster in self.subclusters]
        )

    def _set_params(self, params: DPMSCParams) -> None:
        if params.cluster is not None:
            self.clusters._set_params(params.cluster)
            for i, subcluster in enumerate(self.subclusters):
                subcluster._set_params(params.subcluster[i])
            super()._set_params(params)
            self.n_components = self.clusters.n_components

    def _get_params_prior(self) -> Any:
        return self.clusters._get_params_prior()

    def _set_params_prior(self, params: Any) -> None:
        if params is not None:
            self.clusters._set_params_prior(params)
            for subcluster in self.subclusters:
                subcluster.prior_dir = DirPrior.from_params(1.0 / 2.0)
                subcluster.prior_nw = self.clusters.prior_nw
