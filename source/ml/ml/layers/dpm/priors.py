from dataclasses import dataclass
from typing import Optional

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

    pi_prior: "DirichletPrior"
    mus_covs_prior: "NIWPrior"

    def __init__(
            self,
            kappa: float, nu: float, sigma_scale: float, prior_sigma_choice='data_std'
    ) -> None:
        super().__init__()
        self.prior_sigma_choice = prior_sigma_choice
        self.kappa = torch.tensor(kappa, dtype=torch.float)
        self.nu = torch.tensor(nu, dtype=torch.float)
        self.sigma_scale = sigma_scale

    def init_priors(self, samples):
        mu_0 = torch.mean(samples, dim=0)

        if self.prior_sigma_choice == 'data_std':
            psi = torch.diag(samples.std(dim=0)) * self.sigma_scale
        elif self.prior_sigma_choice == 'isotropic':
            psi = torch.eye(samples.shape[1]) * self.sigma_scale
        else:
            raise ValueError(f"Invalid prior_sigma_choice={self.prior_sigma_choice}")

        self.pi_prior = DirichletPrior()
        self.mus_covs_prior = NIWPrior(self.kappa, self.nu, mu_0, psi)

    def compute_post_params(
            self, D: float, N_K: Tensor,
            data_pi: Tensor, data_mus: Tensor, data_covs: Tensor,
            mus: Optional[Tensor] = None
    ):
        pi_new = self.pi_prior.compute_post_params(data_pi)
        mus_new, covs_new = self.mus_covs_prior.compute_post_params(D, N_K, data_mus, data_covs, mus)
        return pi_new, mus_new, covs_new

    def log_marginal_likelihood(self, samples_k, mu_k):
        return self.mus_covs_prior.log_marginal_likelihood(samples_k, mu_k)


@dataclass
class DirichletPrior:
    def __init__(self) -> None:
        super().__init__()

    def compute_post_params(self, pi) -> Tensor:
        counts = 0.1
        return (pi + counts) / (pi + counts).sum()


@dataclass
class NIWPrior:
    """
    See supplementary material of Sec 3.
    """
    kappa: Tensor
    nu: Tensor
    mu_0: Tensor
    psi: Tensor

    def compute_post(self, N_K: Tensor, data_mus: Tensor, data_covs: Tensor, mus: Optional[Tensor] = None):
        """
        :param N_K: Number of points in each cluster (sum of responses in soft assignment)
        :param data_mus: Data mean
        :param data_covs: Data covariance
        :param mus: Cluster means if unapplicable, then data_mus is used
        :return:
        """
        N_K = N_K.reshape(-1, 1)
        mus = data_mus if mus is None else mus

        # Compute posterior mean
        kappas_star = self.kappa + N_K  # Eq. (13)
        nus_star = self.nu + N_K  # Eq. (15)
        mus_0_star = ((self.kappa * self.mu_0) + (N_K * data_mus)) / kappas_star  # Eq. (14)
        psis_star = (
                self.psi
                + (N_K.unsqueeze(2) * data_covs)
                + (self.kappa * N_K / kappas_star).unsqueeze(2)
                * (mus - self.mu_0).unsqueeze(2) @ (mus - self.mu_0).unsqueeze(1)
        )  # Eq. (16 and 14)

        return kappas_star, nus_star, mus_0_star, psis_star

    def compute_post_params(self, D: float, N_K: Tensor, data_mus: Tensor, data_covs: Tensor,
                            mus: Optional[Tensor] = None):
        """
        :param D: Dimensionality of data
        :param N_K: Number of points in each cluster (sum of responses in soft assignment)
        :param data_mus: Data mean
        :param data_covs: Data covariance
        :param mus: Cluster means if unapplicable, then data_mus is used
        :return:
        """
        kappas_star, nus_star, mus_0_star, psis_star = self.compute_post(N_K, data_mus, data_covs, mus)
        mus_new = mus_0_star  # Eq. (18)
        covs_new = psis_star / (nus_star.unsqueeze(2) + D + 2)  # Eq. (19)
        return mus_new, covs_new

    def log_marginal_likelihood(self, samples_k: Tensor, mu_k: Tensor):
        # TODO: accept mu and cov as parameters instead
        (_, D) = samples_k.shape
        N_K = torch.tensor([len(samples_k)])

        mu_K = mu_k.unsqueeze(0)
        data_mu = torch.mean(samples_k, dim=0).unsqueeze(0)
        samples_minus_mu = (samples_k - mu_k) / len(samples_k)
        data_cov = (samples_minus_mu.T @ samples_minus_mu).unsqueeze(0)

        kappas_star, nus_star, mus_0_star, psis_star = self.compute_post(N_K, data_mu, data_cov, mu_K)
        return (
                - (N_K * D / 2.0) * np.log(np.pi)
                + mvlgamma(nus_star / 2.0, D)
                - mvlgamma(self.nu / 2.0, D)
                + (self.nu / 2.0) * torch.logdet(self.psi)
                - (nus_star / 2.0) * torch.logdet(psis_star)
                + (D / 2.0) * (np.log(self.kappa) - np.log(kappas_star))
        )  # Log variant of Eq. (17)
