import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from ml.layers.clustering import KMeans
from ml.layers.dpm import initialize_kmeans, Priors, initialize_kmeans1d, initialize_soft_assignment, \
    update_cluster_params


def eps_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x + eps) / (x + eps).sum(dim=-1, keepdim=True)


class GMMModule(torch.nn.Module):
    mus: torch.nn.Parameter
    covs: torch.nn.Parameter
    pi: torch.nn.Parameter

    def __init__(
            self, n_components: int, repr_dim: int,
            sim='euclidean',
            mu_init=None, covs_init=None
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.repr_dim = repr_dim
        self.sim = sim

        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')

        self._init_params(mu_init, covs_init)

    def _init_params(self, mu_init, covs_init):
        if mu_init is None:
            mu_init = torch.randn(self.n_components, self.repr_dim)
        assert mu_init.shape == (self.n_components, self.repr_dim)
        self.mus = torch.nn.Parameter(mu_init, requires_grad=False)

        if covs_init is None:
            covs_init = torch.eye(self.repr_dim).reshape(1, self.repr_dim, self.repr_dim).repeat(self.n_components, 1, 1)
        assert covs_init.shape == (self.n_components, self.repr_dim, self.repr_dim)
        self.covs = torch.nn.Parameter(covs_init, requires_grad=False)

        self.pi = torch.nn.Parameter(torch.Tensor(self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)

    def initialize(self, X: Tensor, r: Tensor, prior: Priors, mode='kmeans'):
        if mode == 'kmeans':
            mus, covs, pi = initialize_kmeans(X, self.n_components, prior, sim=self.sim)
        elif mode == 'kmeans1d':
            mus, covs, pi = initialize_kmeans1d(X, self.n_components, prior, sim=self.sim)
        elif mode == 'soft_assignment':
            mus, covs, pi = initialize_soft_assignment(X, r, self.n_components, prior)
        else:
            raise NotImplementedError

        self.mus.data, self.covs.data, self.pi.data = mus, covs, pi

    def estimate_log_prob(self, x: Tensor):
        weighted_r_E = []
        for k in range(self.n_components):
            gmm_k = MultivariateNormal(self.mus[k], self.covs[k])
            prob_k = gmm_k.log_prob(x.detach())
            weighted_r_E.append(prob_k + torch.log(self.pi[k]))

        weighted_r_E = torch.stack(weighted_r_E, dim=1)
        max_values, _ = weighted_r_E.max(dim=1, keepdim=True)
        # r_E_norm = torch.log(torch.sum(torch.exp(weighted_r_E - max_values), dim=1, keepdim=True)) + max_values
        r_E_norm = torch.logsumexp(weighted_r_E - max_values, dim=1, keepdim=True) + max_values
        r_E = torch.exp(weighted_r_E - r_E_norm)

        return r_E

    def e_step(self, x: Tensor, r: Tensor):
        r = eps_norm(r)
        r_E = eps_norm(self.estimate_log_prob(x))

        loss = self.kl_div(torch.log(r), r_E)

        return loss

    def m_step(self, x: Tensor, r: Tensor, prior: Priors):
        mus, covs, pi = update_cluster_params(x, r, self.n_components, prior)

        self.mus.data, self.covs.data, self.pi.data = mus, covs, pi
