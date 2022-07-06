from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, TypeVar, Generic, List

import torch
from torch import Tensor

from ml.algo.dpmm.statistics import InitMode, GaussianParams
from ml.utils import Metric, HParams


class EMCallback:
    def on_after_init(self, model: 'BaseMixture') -> None:
        pass

    def on_after_init_params(self, model: 'BaseMixture') -> None:
        pass

    def on_before_step(self, model: 'BaseMixture') -> None:
        pass

    def on_after_step(self, model: 'BaseMixture', lower_bound: Tensor) -> None:
        pass

    def on_done(self, model: 'BaseMixture', params: Any, i: int) -> None:
        pass

    def on_improvement(self, model: 'BaseMixture', params: Any) -> None:
        pass


class EMAggCallback(EMCallback):
    def __init__(self, callbacks: List[EMCallback]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def on_after_init(self, model: 'BaseMixture') -> None:
        for callback in self.callbacks:
            callback.on_after_init(model)

    def on_after_init_params(self, model: 'BaseMixture') -> None:
        for callback in self.callbacks:
            callback.on_after_init_params(model)

    def on_before_step(self, model: 'BaseMixture') -> None:
        for callback in self.callbacks:
            callback.on_before_step(model)

    def on_after_step(self, model: 'BaseMixture', lower_bound: Tensor) -> None:
        for callback in self.callbacks:
            callback.on_after_step(model, lower_bound)

    def on_done(self, model: 'BaseMixture', params: Any, i) -> None:
        for callback in self.callbacks:
            callback.on_done(model, params, i)

    def on_improvement(self, model: 'BaseMixture', params: Any) -> None:
        for callback in self.callbacks:
            callback.on_improvement(model, params)


@dataclass
class MixtureParams(HParams):
    init_k: int = 2
    reg_cov: float = 1e-6
    init_mode: InitMode = InitMode.KMEANS
    metric: Metric = Metric.DOTP
    tol: float = 1e-4


P = TypeVar('P')


class BaseMixture(Generic[P]):
    params: P = None

    def __init__(self, hparams: MixtureParams) -> None:
        super().__init__()
        self.hparams = hparams
        self.n_components = hparams.init_k
        self.is_fitted = False

    @property
    def inited(self):
        return self.params is not None

    @abstractmethod
    def _init(self, X: Tensor) -> None:
        pass

    @abstractmethod
    def _init_params(self, X: Tensor, z_init: Tensor = None) -> None:
        pass

    def fit(
        self,
        X: Tensor,
        n_init: int = 1, max_iter: int = 100,
        incremental: bool = False,
        callbacks: List[EMCallback] = None,
        z_init: Tensor = None,
    ) -> None:
        assert not incremental or self.params is not None, "Incremental fit requires initialized params"
        callback = EMAggCallback(callbacks or [])

        self._init(X)
        callback.on_after_init(self)

        max_lower_bound = -torch.inf
        best_params = None
        converged = False

        initial_params = self._get_params()
        for i in range(n_init):
            if incremental:
                self._set_params(initial_params)
            else:
                self._init_params(X, z_init=z_init)
            callback.on_after_init_params(self)

            lower_bound = -torch.inf
            j = 0
            for j in range(1, max_iter + 1):
                prev_lower_bound = lower_bound

                callback.on_before_step(self)
                _, log_r = self._e_step(X)
                self._m_step(X, log_r)
                lower_bound = self._compute_lower_bound(X, log_r)
                callback.on_after_step(self, lower_bound)

                change = lower_bound - prev_lower_bound
                if abs(change) < self.hparams.tol:
                    converged = self._on_converge(X, log_r)
                    if converged:
                        break

            if lower_bound > max_lower_bound or max_lower_bound == -torch.inf:
                max_lower_bound = lower_bound
                best_params = self._get_params()
                callback.on_improvement(self, best_params)

            callback.on_done(self, self._get_params(), j)

        self._set_params(best_params)  # TODO: subcluster params shouldnt count here
        self.is_fitted = True

    def predict(self, X: Tensor) -> Tensor:
        return self._estimate_weighted_log_prob(X).argmax(dim=1)

    def _e_step(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        log_prob_norm, log_prob = self._estimate_log_prob_resp(X)
        return log_prob_norm.mean(), log_prob

    @abstractmethod
    def _m_step(self, X: Tensor, log_r: Tensor) -> None:
        pass

    def _on_converge(self, X: Tensor, log_r: Tensor) -> bool:
        return True

    def _estimate_weighted_log_prob(self, X: Tensor) -> Tensor:
        return self._estimate_log_weights() + self._estimate_log_prob(X)

    @abstractmethod
    def _estimate_log_weights(self) -> Tensor:
        pass

    @abstractmethod
    def _estimate_log_prob(self, X: Tensor) -> Tensor:
        pass

    def _estimate_log_prob_resp(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        log_resp = weighted_log_prob - log_prob_norm[:, None]
        return log_prob_norm, log_resp

    def estimate_log_resp(self, X: Tensor) -> Tensor:
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return log_resp

    @abstractmethod
    def _compute_lower_bound(self, X: Tensor, log_r: Tensor) -> Tensor:
        pass

    def _get_params(self) -> P:
        return self.params

    def _set_params(self, params: P) -> None:
        self.params = params

    @property
    @abstractmethod
    def cluster_params(self) -> GaussianParams:
        pass
