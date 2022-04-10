import numpy as np
import torch
from torch import Tensor, mvlgamma


class Priors:
    """
    Priors for MAP inference. See supplementary material of the paper below

    [1] M. Ronen, S. E. Finder, and O. Freifeld,
    “DeepDPM: Deep Clustering With an Unknown Number of Clusters,”
    arXiv:2203.14309 [cs, stat], Mar. 2022, Accessed: Apr. 09, 2022. [Online]. Available: http://arxiv.org/abs/2203.14309
    """
    name = "priors"

    def __init__(
            self,
            K,
            kappa: float, nu: float, sigma_scale: float,
            pi_counts=0.1, counts=10
    ) -> None:
        super().__init__()
        self.pi_prior = DirichletPrior(K, counts)
        self.pi_counts = pi_counts
        self.mus_covs_prior = NIWPrior(kappa, nu, sigma_scale)

    def update_pi_prior(self, K_new, counts=10):
        self.pi_prior = DirichletPrior(K_new, counts)

    def comp_post_counts(self, counts):
        return self.pi_prior.comp_post_counts(counts)

    def compute_post_pi(self, pi):
        return self.pi_prior.comp_post_pi(pi)

    def get_sum_counts(self):
        return self.pi_prior.get_sum_counts()

    def init_priors(self, samples):
        return self.mus_covs_prior.init_priors(samples)

    def compute_params_post(self, samples_k, mu_k):
        return self.mus_covs_prior.compute_params_post(samples_k, mu_k)

    def compute_post_mus(self, N_ks, data_mus):
        return self.mus_covs_prior.compute_post_mus(N_ks, data_mus)

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        return self.mus_covs_prior.compute_post_cov(N_k, mu_k, data_cov_k)

    def log_marginal_likelihood(self, samples_k, mu_k):
        return self.mus_covs_prior.log_marginal_likelihood(samples_k, mu_k)


class DirichletPrior:
    name = "Dirichlet"

    def __init__(self, K: int, counts=10) -> None:
        super().__init__()
        self.K = K
        self.counts = counts
        self.p_counts = torch.ones(K) * counts
        self.pi = self.p_counts / float(K * counts)

    def comp_post_counts(self, counts: Tensor = None) -> Tensor:
        if counts is None:
            counts = self.counts
        return counts + self.p_counts

    def comp_post_pi(self, pi, counts: Tensor = None) -> Tensor:
        if counts is None:
            counts = 0.1
        return (pi + counts) / (pi + counts).sum()

    def get_sum_counts(self):
        return self.K * self.counts


class NIWPrior:
    """
    See supplementary material of Sec 3.
    """
    name = "NIW"

    kappa: float
    nu: float
    m: Tensor
    psi: Tensor

    def __init__(self, kappa: float, nu: float, sigma_scale: float) -> None:
        super().__init__()
        self.sigma_scale = sigma_scale
        self.kappa = kappa
        self.nu = nu

    def init_priors(self, samples: Tensor):
        self.m = torch.mean(samples, dim=0)
        self.psi = torch.eye(samples.shape[1]) * self.sigma_scale
        return self.m, self.psi

    def compute_params_post(self, samples_k: Tensor, mu_k: Tensor):
        # Compute posterior parameters Eq. (12) - Hard Assignment (3.1)
        N_k = len(samples_k)
        kappa_star = self.kappa + N_k  # Eq. (13)
        nu_star = self.nu + N_k  # Eq. (15)
        mu_0_star = (self.kappa * self.m + samples_k.sum(axis=0)) / kappa_star  # Eq. (14)
        samples_minus_mu = samples_k - mu_k
        S = samples_minus_mu.T @ samples_minus_mu
        psi_star = (
                self.psi
                + S
                + (self.kappa * N_k / kappa_star)
                * (mu_k - self.m).unsqueeze(1) @ (mu_k - self.m).unsqueeze(0)
        )  # Eq. (16)

        return kappa_star, nu_star, mu_0_star, psi_star

    def compute_post_mus(self, N_ks, data_mus):
        # N_k is the number of points in cluster K for hard assignment, and the sum of all responses to the K-th cluster for soft assignment
        return (
                ((self.kappa * self.m) + (N_ks.reshape(-1, 1) * data_mus))
                / (N_ks.reshape(-1, 1) + self.kappa)  # Eq. (14)
        )  # Eq. (19)

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)
        D = len(mu_k)
        if N_k > 0:
            return (
               self.psi
               + data_cov_k * N_k  # unnormalize
               + (
                       ((self.kappa * N_k) / (self.kappa + N_k))
                       * ((mu_k - self.m).unsqueeze(1) * (mu_k - self.m).unsqueeze(0))
               )
           ) / (self.nu + N_k + D + 2)  # Eq. (16)
        else:
            return self.psi

    def log_marginal_likelihood(self, samples_k: Tensor, mu_k: Tensor):
        kappa_star, nu_star, mu_0_star, psi_star = self.compute_params_post(samples_k, mu_k)
        (N_k, D) = samples_k.shape
        return (
                - (N_k * D / 2.0) * np.log(np.pi)
                + mvlgamma(torch.tensor(nu_star / 2.0), D)
                - mvlgamma(torch.tensor(self.nu) / 2.0, D)
                + (self.nu / 2.0) * torch.logdet(self.psi)
                - (nu_star / 2.0) * torch.logdet(psi_star)
                + (D / 2.0) * (np.log(self.kappa) - np.log(kappa_star))
        )  # Log variant of Eq. (17)
