"""Regularised horseshoe prior with mean-field variational inference.

Statistical background
----------------------
Carvalho, Polson & Scott (2010) introduced the horseshoe prior for the sparse
normal-means problem:

    β_j | λ_j, τ  ~  N(0, λ_j² τ²)
    λ_j            ~  Half-Cauchy(0, 1)   (local scale)
    τ              ~  Half-Cauchy(0, τ₀)  (global scale)

The shrinkage coefficient κ_j = 1 / (1 + λ_j²) follows a Beta(1/2, 1/2)
distribution — the U-shaped "horseshoe" profile.  Most coefficients are
shrunk fully to zero (κ ≈ 1) while true signals are left unshrunk (κ ≈ 0).

Regularised horseshoe (Piironen & Vehtari 2017)
-----------------------------------------------
The original horseshoe has unbounded tails which cause divergent transitions
in HMC.  The regularised version replaces λ_j with:

    λ̃_j² = c² λ_j² / (c² + τ² λ_j²)

where c² ~ Inv-Gamma(ν_slab/2, ν_slab s²_slab/2) is a finite slab width.
This preserves the U-shaped spike while bounding the unshrunk slab to N(0, c²).

Global scale τ₀ (Piironen & Vehtari 2017)
------------------------------------------
    τ₀ = (p₀ / (D − p₀)) · (σ / √n)

where p₀ is the expected number of nonzero coefficients, D is the total number
of predictors (TFs), σ is the noise scale, and n is the sample size.  For
GRN inference: p₀ ≈ 10–20 (typical TF regulators per gene), D = n_tfs.

Mean-field VI
-------------
We use a mean-field approximate posterior q(β, λ, τ, c²) factored as:

    q(β_j) = N(μ_j, σ_j²)           (posterior edge weights — diagonal)
    q(λ̃_j) = LogNormal(mλ_j, sλ_j²) (local scale — positive)
    q(τ)   = LogNormal(mτ, sτ²)      (global scale — positive)
    q(c²)  = LogNormal(mc, sc²)       (slab width — positive)

The ELBO is: E_q[log p(y|β)] − KL(q || p_horseshoe).  We optimise only the
KL term here; the likelihood term is handled by the JEPA prediction loss in
the outer model.

Implementation note
-------------------
Rather than parameterising λ_j directly (which requires integrating out the
half-Cauchy prior), we use the reparameterisation trick for LogNormal
variational families and estimate the KL via Monte Carlo.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# τ₀ formula
# ---------------------------------------------------------------------------

def compute_tau0(
    p0: float,
    n_tfs: int,
    n_samples: int,
    sigma: float = 1.0,
) -> float:
    """Piironen & Vehtari (2017) recommended global scale.

    Parameters
    ----------
    p0:
        Expected number of nonzero TF regulators per target gene.
    n_tfs:
        Total number of TF predictors (D in the formula).
    n_samples:
        Number of expression experiments (n in the formula).
    sigma:
        Noise standard deviation.  Use 1.0 for standardised expression.

    Returns
    -------
    float
        τ₀ value to use as the scale of the Half-Cauchy prior on τ.
    """
    D = n_tfs
    return (p0 / (D - p0)) * (sigma / math.sqrt(n_samples))


# ---------------------------------------------------------------------------
# Horseshoe regression head
# ---------------------------------------------------------------------------

class HorseshoeRegressor(nn.Module):
    """Per-gene horseshoe Bayesian linear regression over TF latent embeddings.

    Maintains variational parameters for the approximate posterior over the
    regulatory weight matrix W ∈ R^{n_tfs × n_target_genes}.

    Each column W[:, g] is the vector of TF→gene_g regulatory coefficients,
    with independent horseshoe priors.

    Parameters
    ----------
    n_tfs:
        Number of transcription factors (predictor dimension).
    n_target_genes:
        Number of target genes (number of regression problems).
    tau0:
        Prior scale for the global shrinkage parameter τ.  Use
        ``compute_tau0()`` to set this from biological sparsity knowledge.
    nu_slab:
        Degrees of freedom for the slab (Inv-Gamma prior on c²).  Default 4.
    s_slab:
        Scale for the slab (Inv-Gamma prior on c²).  Default 2.
    """

    def __init__(
        self,
        n_tfs: int,
        n_target_genes: int,
        tau0: float,
        nu_slab: float = 4.0,
        s_slab: float = 2.0,
    ) -> None:
        super().__init__()

        self.n_tfs = n_tfs
        self.n_target_genes = n_target_genes
        self.tau0 = tau0
        self.nu_slab = nu_slab
        self.s_slab = s_slab

        # ----------------------------------------------------------------
        # Variational parameters (log-space for positive quantities)
        # ----------------------------------------------------------------

        # Posterior over edge weights β: q(β_jg) = N(μ_jg, exp(log_σ_jg)²)
        #
        # Two critical init choices:
        # 1. mu_W: small random (not zero). When mu_W=0, W.T @ s_context=0
        #    for every gene → zero JEPA gradient into mu_W → no learning.
        # 2. log_sigma_W: initialised so softplus(log_sigma_W) ≈ tau0.
        #    The prior std at init is λ̃·τ ≈ 1·tau0 (with λ̃=1, τ=tau0).
        #    Matching sigma_W to the prior std makes the per-element KL ≈ 0
        #    at init, so the JEPA loss can pull mu_W toward signal before
        #    the horseshoe starts aggressively shrinking weights.
        #    For small tau0: softplus(log(tau0)) ≈ tau0. ✓
        self.mu_W = nn.Parameter(torch.empty(n_tfs, n_target_genes).normal_(0, tau0))
        log_sigma_init = math.log(tau0 + 1e-8)   # softplus(log(tau0)) ≈ tau0
        self.log_sigma_W = nn.Parameter(torch.full((n_tfs, n_target_genes), log_sigma_init))

        # Local scales λ̃_jg: q = LogNormal(m_lambda, exp(log_s_lambda)²)
        self.m_lambda = nn.Parameter(torch.zeros(n_tfs, n_target_genes))
        self.log_s_lambda = nn.Parameter(torch.full((n_tfs, n_target_genes), -1.0))

        # Global scale τ: q = LogNormal(m_tau, exp(log_s_tau)²)
        self.m_tau = nn.Parameter(torch.tensor(math.log(tau0)))
        self.log_s_tau = nn.Parameter(torch.tensor(-1.0))

        # Slab width c²: q = LogNormal(m_c, exp(log_s_c)²)
        self.m_c = nn.Parameter(torch.tensor(math.log(s_slab ** 2)))
        self.log_s_c = nn.Parameter(torch.tensor(-1.0))

    # ----------------------------------------------------------------
    # Sampling with reparameterisation trick
    # ----------------------------------------------------------------

    def _sample_lognormal(
        self, m: torch.Tensor, log_s: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Sample from LogNormal(m, exp(log_s)²) via reparameterisation."""
        s = F.softplus(log_s) + 1e-6
        eps = torch.randn(*m.shape, n_samples, device=m.device)
        return torch.exp(m.unsqueeze(-1) + s.unsqueeze(-1) * eps)

    def sample_W(self, n_mc: int = 1) -> torch.Tensor:
        """Sample regulatory weights from the variational posterior.

        Returns
        -------
        Tensor, shape (n_tfs, n_target_genes, n_mc)
        """
        sigma_W = F.softplus(self.log_sigma_W) + 1e-6
        eps = torch.randn(*self.mu_W.shape, n_mc, device=self.mu_W.device)
        return self.mu_W.unsqueeze(-1) + sigma_W.unsqueeze(-1) * eps

    # ----------------------------------------------------------------
    # KL divergence (regularisation term)
    # ----------------------------------------------------------------

    def kl_divergence(self, n_mc: int = 4) -> torch.Tensor:
        """Estimate KL(q || p_horseshoe) via Monte Carlo.

        Returns a scalar KL divergence averaged over the weight matrix.
        """
        # Sample local scales, global scale, slab width
        lambda_tilde = self._sample_lognormal(self.m_lambda, self.log_s_lambda, n_mc)  # (n_tfs, n_genes, n_mc)
        tau = self._sample_lognormal(self.m_tau.unsqueeze(0).unsqueeze(0), self.log_s_tau.unsqueeze(0).unsqueeze(0), n_mc)  # (1, 1, n_mc)
        c2 = self._sample_lognormal(self.m_c.unsqueeze(0).unsqueeze(0), self.log_s_c.unsqueeze(0).unsqueeze(0), n_mc)   # (1, 1, n_mc)

        # Effective prior variance for each weight: λ̃² τ²
        prior_var = (lambda_tilde ** 2) * (tau ** 2)  # (n_tfs, n_genes, n_mc)

        # KL for the weight posterior q(β) = N(μ, σ²) against p(β) = N(0, prior_var)
        # KL(N(μ,σ²) || N(0,v)) = 0.5 * (σ²/v + μ²/v - 1 + log(v/σ²))
        sigma_W = (F.softplus(self.log_sigma_W) + 1e-6).unsqueeze(-1)   # (n_tfs, n_genes, 1)
        mu_W    = self.mu_W.unsqueeze(-1)                                  # (n_tfs, n_genes, 1)

        kl_w = 0.5 * (
            sigma_W ** 2 / prior_var
            + mu_W ** 2 / prior_var
            - 1.0
            + torch.log(prior_var / (sigma_W ** 2 + 1e-8))
        )  # (n_tfs, n_genes, n_mc)

        # KL for τ: q = LogNormal(m_τ, s_τ²), p = Half-Cauchy(0, τ₀)
        # Approximate: use log-normal entropy + E_q[log p(τ)]
        # Half-Cauchy log-density: log p(τ) ∝ -log(1 + (τ/τ₀)²)
        kl_tau = self._kl_lognormal_halfcauchy(
            self.m_tau, F.softplus(self.log_s_tau) + 1e-6, self.tau0
        )

        # KL for λ̃: q = LogNormal, p = Half-Cauchy(0,1)
        kl_lambda = self._kl_lognormal_halfcauchy(
            self.m_lambda, F.softplus(self.log_s_lambda) + 1e-6, 1.0
        ).mean()

        return kl_w.mean() + kl_tau + kl_lambda

    @staticmethod
    def _kl_lognormal_halfcauchy(
        m: torch.Tensor, s: torch.Tensor, scale: float, n_mc: int = 16
    ) -> torch.Tensor:
        """MC estimate of KL(LogNormal(m,s²) || HalfCauchy(0, scale))."""
        eps = torch.randn(n_mc, device=m.device)
        # Sample from q: x = exp(m + s * ε)
        x = torch.exp(m.mean() + s.mean() * eps)
        # log q(x) for LogNormal
        log_q = -torch.log(x) - torch.log(s.mean()) - 0.5 * ((torch.log(x) - m.mean()) / s.mean()) ** 2
        # log p(x) for Half-Cauchy: log p(x) = log(2/(π scale)) - log(1 + (x/scale)²)
        log_p = math.log(2 / (math.pi * scale)) - torch.log(1.0 + (x / scale) ** 2)
        return (log_q - log_p).mean()

    # ----------------------------------------------------------------
    # Posterior summary
    # ----------------------------------------------------------------

    @torch.no_grad()
    def posterior_mean_W(self) -> torch.Tensor:
        """Return the variational posterior mean of W.

        Shape: (n_tfs, n_target_genes)
        This is the quantity used as input to the JEPAPredictor.
        """
        return self.mu_W.clone()

    @torch.no_grad()
    def posterior_std_W(self) -> torch.Tensor:
        """Return the variational posterior std of W (edge-level uncertainty).

        Shape: (n_tfs, n_target_genes)
        """
        return F.softplus(self.log_sigma_W) + 1e-6

    @torch.no_grad()
    def edge_scores(self) -> torch.Tensor:
        """Signed edge score: posterior mean (used for GRN ranking).

        Shape: (n_tfs, n_target_genes)
        Absolute value gives confidence; sign gives activation/repression.
        """
        return self.mu_W.clone()

    @torch.no_grad()
    def shrinkage_weights(self) -> torch.Tensor:
        """Estimated shrinkage κ_j = 1/(1 + λ̃_j²) ∈ (0,1).

        κ ≈ 1 → fully shrunk (non-edge).
        κ ≈ 0 → unshrunk (true regulatory edge).

        Shape: (n_tfs, n_target_genes)
        """
        lambda_tilde = torch.exp(self.m_lambda)
        return 1.0 / (1.0 + lambda_tilde ** 2)
