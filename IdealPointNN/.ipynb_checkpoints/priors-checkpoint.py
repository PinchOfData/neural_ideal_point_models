#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from sklearn.linear_model import MultiTaskElasticNetCV


class Prior:
    """
    Base template class for priors.
    """

    def __init__(self):
        pass

    def update_parameters(self):
        """
        M-step after each epoch.
        """
        pass

    def sample(self):
        """
        Sample from the prior.
        """
        pass

    def simulate(self):
        """
        Simulate data to test the prior's updating rule.
        """
        pass


class NormalPrior(Prior):
    """
    Normal prior
    """

    def __init__(
        self,
        prevalence_covariate_size,
        n_dims,
        device,
    ):
        self.prevalence_covariates_size = prevalence_covariate_size
        self.n_dims = n_dims
        self.device = device
        if prevalence_covariate_size != 0:
            self.lambda_ = torch.zeros(prevalence_covariate_size, n_dims).to(
                self.device
            )
            self.sigma = torch.diag(torch.Tensor([1.0] * self.n_dims)).to(self.device)

    def update_parameters(self, posterior_mu, M_prevalence_covariates, **kwargs):
        """
        M-step after each epoch.
        """

        reg = MultiTaskElasticNetCV(fit_intercept=False, **kwargs)
        reg.fit(M_prevalence_covariates, posterior_mu)
        lambda_ = reg.coef_
        self.lambda_ = torch.from_numpy(lambda_.T).to(self.device)

        posterior_mu = torch.from_numpy(posterior_mu).to(self.device)
        M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(
            self.device
        )
        difference_in_means = posterior_mu - torch.matmul(
            M_prevalence_covariates, self.lambda_
        )
        self.sigma = (
            torch.matmul(difference_in_means.T, difference_in_means)
            / posterior_mu.shape[0]
        )

    def sample(self, N, M_prevalence_covariates):
        """
        Sample from the prior.
        """
        if self.prevalence_covariates_size == 0:
            z_true = np.random.randn(N, self.n_dims)
            z_true = torch.from_numpy(z_true).to(
                    self.device
                )
        else:
            if torch.is_tensor(M_prevalence_covariates) == False:
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(
                    self.device
                )
            means = torch.matmul(M_prevalence_covariates, self.lambda_)
            for i in range(means.shape[0]):
                if i == 0:
                    m = MultivariateNormal(means[i], self.sigma)
                    z_true = m.sample().unsqueeze(0)
                else:
                    m = MultivariateNormal(means[i], self.sigma)
                    z_temp = m.sample()
                    z_true = torch.cat([z_true, z_temp.unsqueeze(0)], 0)
        return z_true.float()

    def simulate(self, M_prevalence_covariates, lambda_, sigma):
        """
        Simulate data to test the prior's updating rule.
        """
        means = torch.matmul(M_prevalence_covariates, lambda_)
        for i in range(means.shape[0]):
            if i == 0:
                m = MultivariateNormal(means[i], sigma)
                z_sim = m.sample().unsqueeze(0)
            else:
                m = MultivariateNormal(means[i], sigma)
                z_temp = m.sample()
                z_sim = torch.cat([z_sim, z_temp.unsqueeze(0)], 0)
        return z_sim.float()

    def get_dim_correlations(self):
        """
        Plot correlations between dimensions for a normal prior.
        """
        # Represent as a standard variance-covariance matrix
        # See https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
        sigma = pd.DataFrame(self.sigma.detach().cpu().numpy())
        mask = np.zeros_like(sigma, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sigma[mask] = np.nan
        p = (
            sigma.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1)
            .highlight_null(color="#f1f1f1")  # Color NaNs grey
            .format(precision=2)
        )
        return p

    def to(self, device):
        """
        Move the model to a different device.
        """
        self.device = device
        self.lambda_ = self.lambda_.to(device)
        self.sigma = self.sigma.to(device)
