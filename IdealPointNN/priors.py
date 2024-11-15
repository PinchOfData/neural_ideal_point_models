#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression,RidgeCV,MultiTaskLassoCV,MultiTaskElasticNetCV
from torch.distributions.dirichlet import Dirichlet


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


class LogisticNormalPrior(Prior):
    """
    Logistic Normal prior

    We draw from a multivariate gaussian and map it to the simplex.
    Does not induce sparsity, but may account for topic correlations.

    References:
        - Roberts, M. E., Stewart, B. M., & Airoldi, E. M. (2016). A model of text for experimentation in the social sciences. Journal of the American Statistical Association, 111(515), 988-1003.
    """

    def __init__(
        self,
        n_topics,
        prevalence_covariate_size=0,
        model_type="RidgeCV",
        prevalence_model_args={},
        device="cuda",
    ):
        self.prevalence_covariates_size = prevalence_covariate_size
        self.n_topics = n_topics
        self.model_type = model_type
        self.prevalence_model_args = prevalence_model_args
        self.device = device
        if prevalence_covariate_size != 0:
            self.lambda_ = torch.zeros(prevalence_covariate_size, n_topics).to(
                self.device
            )
            self.sigma = torch.diag(torch.Tensor([1.0] * self.n_topics)).to(self.device)

    def update_parameters(self, posterior_mu, M_prevalence_covariates):
        """
        M-step after each epoch.
        """
        if self.model_type == "MultiTaskElasticNetCV":
            reg = MultiTaskElasticNetCV(fit_intercept=False, **self.prevalence_model_args)
        elif self.model_type == "MultiTaskLassoCV":
            reg = MultiTaskLassoCV(fit_intercept=False, **self.prevalence_model_args)
        elif self.model_type == "RidgeCV":
            reg = RidgeCV(fit_intercept=False, **self.prevalence_model_args)
        else:
            reg = LinearRegression(fit_intercept=False, **self.prevalence_model_args)
            
        reg.fit(M_prevalence_covariates, posterior_mu)
        lambda_ = reg.coef_
        self.lambda_ = torch.from_numpy(lambda_.T).to(self.device)

        posterior_mu = torch.from_numpy(posterior_mu).to(self.device)
        M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(
            self.device
        )
        difference_in_means = posterior_mu - torch.matmul(
            M_prevalence_covariates, self.lambda_.to(torch.float32)
        )
        self.sigma = (
            torch.matmul(difference_in_means.T, difference_in_means)
            / posterior_mu.shape[0]
        )

        self.lambda_ = self.lambda_ - self.lambda_[:, 0][:, None]
        self.lambda_ = self.lambda_.to(torch.float32)

    def sample(self, N, M_prevalence_covariates=None, to_simplex=True, epoch=None, initialization=False):
        """
        Sample from the prior.
        """
        if self.prevalence_covariates_size == 0 or initialization:
            z_true = np.random.randn(N, self.n_topics)
            z_true = torch.from_numpy(z_true).to(
                    self.device
                )
        else:
            if torch.is_tensor(M_prevalence_covariates) == False:
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(
                    self.device
                )
            means = torch.matmul(M_prevalence_covariates, self.lambda_)       
            z_true = torch.empty((means.shape[0], self.sigma.shape[0]))
            for i in range(means.shape[0]):
                m = MultivariateNormal(means[i], self.sigma)
                z_true[i] = m.sample()
        if to_simplex:
            z_true = torch.softmax(z_true, dim=1)
        return z_true.float()

    def simulate(self, M_prevalence_covariates, lambda_, sigma, to_simplex=False):
        """
        Simulate data to test the prior's updating rule.
        """
        means = torch.matmul(M_prevalence_covariates, lambda_)
        z_sim = torch.empty((means.shape[0], sigma.shape[0]))
        for i in range(means.shape[0]):
            m = MultivariateNormal(means[i], sigma)
            z_sim[i] = m.sample()
        if to_simplex:
            z_sim = torch.softmax(z_sim, dim=1)
        return z_sim.float()

    def get_topic_correlations(self):
        """
        Plot correlations between topics for a logistic normal prior.
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


class DirichletPrior(Prior):
    """
    Dirichlet prior

    Induces sparsity, but does not account for topic correlations.

    References:
        - Mimno, D. M., & McCallum, A. (2008, July). Topic models conditioned on arbitrary features with Dirichlet-multinomial regression. In UAI (Vol. 24, pp. 411-418).
        - Maier, M. (2014). DirichletReg: Dirichlet regression for compositional data in R.
    """

    def __init__(
        self,
        n_topics,
        alpha,
        device,
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.device = device
        self.lambda_ = None

    def sample(self, N):
        """
        Sample from the prior.
        """

        z_true = np.random.dirichlet(np.ones(self.n_topics) * self.alpha, size=N)
        z_true = torch.from_numpy(z_true).float()

        return z_true

    def to(self, device):
        """
        Move the model to a different device.
        """
        self.device = device
        self.linear_model = self.linear_model.to(device)