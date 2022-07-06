import unittest

import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _estimate_log_gaussian_prob, \
    _compute_precision_cholesky

from ml.algo.dpmm.dpmm import DPMM, DPMMParams
from ml.algo.dpmm.prior import DirPrior, NWPrior
from ml.algo.dpmm.statistics import estimate_gaussian_parameters, covs_to_prec, estimate_gaussian_log_prob


class TestDPMM(unittest.TestCase):
    def test_gauss_params(self):
        X = torch.rand(100, 4)
        r = torch.rand(100, 5)
        r /= r.sum(dim=1, keepdim=True)

        Ns, mus, covs = _estimate_gaussian_parameters(X.numpy(), r.numpy(), reg_covar=1e-6, covariance_type="full")
        Ns_, mus_, covs_ = estimate_gaussian_parameters(X, r, reg_covar=1e-6)
        self.assertTrue(torch.allclose(torch.from_numpy(Ns).float(), Ns_))
        self.assertTrue(torch.allclose(torch.from_numpy(mus).float(), mus_))
        self.assertTrue(torch.allclose(torch.from_numpy(covs).float(), covs_))

        precs = _compute_precision_cholesky(covs, "full")
        precs_ = covs_to_prec(covs_)
        self.assertTrue(torch.allclose(torch.from_numpy(precs).float(), precs_, rtol=0.00001, atol=0.00001))

        prob = _estimate_log_gaussian_prob(X.numpy(), mus, precs, "full")
        prob_ = estimate_gaussian_log_prob(X, mus_, torch.from_numpy(precs).float())
        self.assertTrue(torch.allclose(torch.from_numpy(prob).float(), prob_, rtol=0.00001, atol=0.00001))

    def test_prior(self):
        D = 4
        k = 5
        X = torch.rand(100, D)
        r = torch.rand(100, k)
        r /= r.sum(dim=1, keepdim=True)

        Ns, mus, covs = estimate_gaussian_parameters(X, r, reg_covar=1e-6)

        prior_alpha = 1.0
        prior_kappa = 1.0
        prior_nu = 5

        bgmm = BayesianGaussianMixture(
            n_components=5,
            covariance_type="full",
            reg_covar=1e-6,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=prior_alpha,
            mean_precision_prior=prior_kappa,
            degrees_of_freedom_prior=prior_nu,
        )
        bgmm._check_parameters(X.numpy())
        bgmm._initialize(X.numpy(), r.numpy())

        prior_pi = DirPrior.from_params(prior_alpha)
        mu, cov = torch.mean(X, dim=0), torch.cov(X.T)
        prior_mu_cov = NWPrior.from_params(prior_kappa, prior_nu, mu, cov, prior_cov_scale=1.0)

        # params_pi_, params_mu_cov = prior_pi.estimate_post(Ns), prior_mu_cov.estimate_post(Ns, mus, covs)

        alpha, alpha_ = bgmm.weight_concentration_prior_, prior_pi.alpha
        self.assertTrue(torch.allclose(torch.tensor(alpha).float(), alpha_))
        mu_0, mu_0_ = bgmm.mean_prior_, prior_mu_cov.mu_0
        self.assertTrue(torch.allclose(torch.tensor(mu_0).float(), mu_0_))
        kappa, kappa_ = bgmm.mean_precision_prior_, prior_mu_cov.kappa
        self.assertTrue(torch.allclose(torch.tensor(kappa).float(), kappa_))
        nu, nu_ = bgmm.degrees_of_freedom_prior_, prior_mu_cov.nu
        self.assertTrue(torch.allclose(torch.tensor(nu).long(), nu_))
        cov_p, cov_p_ = bgmm.covariance_prior_, prior_mu_cov.W_inv
        self.assertTrue(torch.allclose(torch.tensor(cov_p).float(), cov_p_))

        params_pi, params_pi_ = bgmm.weight_concentration_, prior_pi.estimate_post(Ns)
        self.assertTrue(torch.allclose(torch.tensor(params_pi[0]).float(), params_pi_[0]))
        self.assertTrue(torch.allclose(torch.tensor(params_pi[1]).float(), params_pi_[1]))

        params_ = prior_mu_cov.estimate_post(Ns, mus, covs)
        (par_mus_, par_kappas_, par_nus_, par_Ws_, par_covs_) = params_
        _, par_kappas, par_mus, par_nus, par_covs, par_Ws = bgmm._get_parameters()
        self.assertTrue(torch.allclose(torch.tensor(par_mus).float(), par_mus_))
        self.assertTrue(torch.allclose(torch.tensor(par_kappas).float(), par_kappas_))
        self.assertTrue(torch.allclose(torch.tensor(par_nus).float(), par_nus_))
        self.assertTrue(torch.allclose(torch.tensor(par_covs).float(), par_covs_, rtol=0.0001, atol=0.0001))
        self.assertTrue(torch.allclose(torch.tensor(par_Ws).float(), par_Ws_, rtol=0.001, atol=0.001))

        pi, pi_ = bgmm._estimate_log_weights(), prior_pi.estimate_log_prob(params_pi_)
        self.assertTrue(torch.allclose(torch.tensor(pi).float(), pi_))

        lc, lc_ = bgmm._estimate_log_prob(X.numpy()), prior_mu_cov.estimate_log_prob(X, params_)
        self.assertTrue(torch.allclose(torch.tensor(lc).float(), lc_, rtol=0.001, atol=0.001))

        dpmm = DPMM(
            n_components=k,
            reg_cov=1e-6,
            prior_alpha=prior_alpha,
            prior_kappa=prior_kappa,
            prior_nu=prior_nu,
        )
        dpmm.prior_pi = prior_pi
        dpmm.prior_mu_cov = prior_mu_cov
        dpmm.params = DPMMParams(params_pi_, params_)

        (lrn, lr), (lrn_, lr_) = bgmm._estimate_log_prob_resp(X.numpy()), dpmm._estimate_log_prob_resp(X)
        self.assertTrue(torch.allclose(torch.tensor(lr).float(), lr_, rtol=0.001, atol=0.001))
        self.assertTrue(torch.allclose(torch.tensor(lrn).float(), lrn_, rtol=0.0001, atol=0.0001))

        lb, lb_ = bgmm._compute_lower_bound(lr, lrn), dpmm._compute_lower_bound(X, lr_)
        lb_ = lb_ - (np.log(torch.pi) * D * (D - 1) / 4) * k  # I use mvlgamma and dont skip the constant
        self.assertTrue(torch.allclose(torch.tensor(lb).float(), lb_, rtol=0.0001, atol=0.0001))

        # bgmm._estimate_weights(Ns.numpy())
        # Ns_post = bgmm.weight_concentration_
