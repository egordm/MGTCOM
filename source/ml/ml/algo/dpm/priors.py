from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

from ml.algo.dpm.statistics import MultivarNormalParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)

Float = Union[float, Tensor]

DirichletParams = NamedTuple('DirichletParams', [('alpha', Float)])


@dataclass
class DirichletPrior:
    params: DirichletParams

    @staticmethod
    def from_params(alpha: float = 1.0) -> Self:
        """
        `from_params` takes a single float as an argument and returns a `DirichletPrior` object

        :param alpha: The concentration parameter for the Dirichlet distribution
        :type alpha: float
        :return: A DirichletPrior object with a DirichletParams object as its attribute.
        """
        return DirichletPrior(DirichletParams(torch.tensor(alpha)))

    def compute_posterior(self, Ns: Tensor) -> Tensor:
        """
        > The posterior probability of a topic is the number of times it appears in a document plus the prior probability of
        the topic, divided by the total number of words in the document plus the prior probability of the topic

        :param Ns: the number of times each topic appears in the corpus
        :type Ns: torch.Tensor
        :return: The posterior probability of the topics given their document counts.
        """
        return (Ns + self.params.alpha) / (Ns.sum() + self.params.alpha)


NIWPriorParams = NamedTuple('NIWPriorParams', [('nu', Float), ('kappa', Float), ('mu', Tensor), ('psi', Tensor)])
NIWPriorParamsList = NamedTuple('NIWPriorParamsList',
                                [('nus', Float), ('kappas', Float), ('mus', Tensor), ('psis', Tensor)])


@dataclass
class NIWPrior:
    params: NIWPriorParams

    @staticmethod
    def from_params(mu: Tensor, psi: Tensor, nu: Float = 12.0, kappa: Float = 0.0001) -> Self:
        """
        `NIWPrior.from_params(mu, psi, nu, kappa)` returns a `NIWPrior` object with the parameters `mu`, `psi`, `nu`, and
        `kappa`

        :param mu: The mean of the distribution
        :type mu: Tensor
        :param psi: The precision matrix (inverse of the covariance matrix)
        :type psi: Tensor
        :param nu: The degrees of freedom of the distribution
        :type nu: Float
        :param kappa: The precision of the mean
        :type kappa: Float
        :return: A NIWPrior object
        """
        D = mu.shape[-1]
        if nu < D:
            logger.warning(f"nu must be at least D + 1. Setting nu to {D + 1}")
            nu = D + 1

        return NIWPrior(NIWPriorParams(
            torch.tensor(nu),
            torch.tensor(kappa),
            mu, psi,
        ))

    def update(self, X: Tensor, sigma_scale: float = 0.005):
        """
        > Update the parameters of the prior distribution with the mean and standard deviation of the data

        :param X: Input data points
        :type X: Tensor
        :param sigma_scale: This is a hyperparameter that controls the variance of the prior
        :type sigma_scale: float
        """
        mu = X.mean(dim=0)
        psi = (torch.diag(X.std(dim=0)) * sigma_scale)

        self.params = NIWPriorParams(
            self.params.nu,
            self.params.kappa,
            mu, psi,
        )

    def compute_posterior(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> NIWPriorParamsList:
        """
        > We update the parameters of the prior by adding the number of data points, the mean, and the covariance of the
        data points to the prior parameters

        :param Ns: number of data points in each topic
        :type Ns: Tensor
        :param mus: the mean of the data points in each topic
        :type mus: Tensor
        :param covs: the covariance of the data points in each topic
        :type covs: Tensor
        :return: The posterior parameters of the NIW prior.
        """
        # mu = X.mean(dim=0)
        # cov = compute_cov(X, mu)
        Ns = Ns.reshape(-1, 1)

        kappas_post = self.params.kappa + Ns
        nus_post = self.params.nu + Ns
        mus_post = (self.params.kappa * self.params.mu + Ns * mus) / kappas_post
        S = covs * Ns.unsqueeze(2)
        psis_post = (
                self.params.psi
                + S
                + ((self.params.kappa * Ns) / kappas_post).unsqueeze(2)
                * (mus - self.params.mu).unsqueeze(2) @ (mus - self.params.mu).unsqueeze(1)
        )

        return NIWPriorParamsList(nus_post, kappas_post, mus_post, psis_post)

    def compute_posterior_mv(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> MultivarNormalParams:
        """
        > Given the number of samples, the mean, and the covariance of each topic, compute the posterior parameters of
        the multivariate normal distribution

        :param Ns: the number of data points in each topic
        :type Ns: Tensor
        :param mus: the mean of the data points in each topic
        :type mus: Tensor
        :param covs: the covariance matrices of the topics
        :type covs: Tensor
        :return: The posterior parameters of the multivariate normal distribution.
        """
        D = mus.shape[-1]
        nus_post, kappas_post, mus_post, psis_post = self.compute_posterior(Ns, mus, covs)
        covs_post = torch.stack([
            psis_post[i] / (nus_post[i] - D + 1) if N_k > 0 else self.params.psi
            for i, N_k in enumerate(Ns)
        ], dim=0)

        return MultivarNormalParams(mus_post, covs_post)

    def marginal_log_prob(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> Tensor:
        """
        $$
        \log p(N, \mu, \Sigma) = \log p(N) + \log p(\mu | N) + \log p(\Sigma | N)
        $$

        where $p(N)$ is the marginal probability of the number of data points, $p(\mu | N)$ is the marginal probability of
        the mean given the number of data points, and $p(\Sigma | N)$ is the marginal probability of the covariance given
        the number of data points

        :param Ns: Number of samples
        :type Ns: Tensor
        :param mus: The mean of the data points
        :type mus: Tensor
        :param covs: The covariance of the data points
        :type covs: Tensor
        :return: The marginal log probability of the data.
        """
        D = mus.shape[-1]
        nus_post, kappas_post, mus_post, psis_post = self.compute_posterior(Ns, mus, covs)

        # TODO: Compare
        return (
                -((Ns * D / 2) * np.log(torch.pi)).reshape(-1, 1)
                + torch.mvlgamma(nus_post / 2.0, D)
                - torch.mvlgamma(self.params.nu / 2.0, D)
                + (self.params.nu / 2.0) * torch.logdet(self.params.psi)
                - (nus_post / 2.0) * torch.logdet(psis_post).reshape(-1, 1)
                + (D / 2.0) * (torch.log(self.params.kappa) - torch.log(kappas_post))
        )
