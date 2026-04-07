"""Regularised horseshoe prior via PyMC + NUTS.

Implements the Piironen & Vehtari (2017) regularised horseshoe exactly:

    β_j | λ̃_j, τ  ~  N(0, λ̃_j² τ²)
    λ̃_j²          =  c² λ_j² / (c² + τ² λ_j²)   [slab regularisation]
    λ_j            ~  HalfCauchy(0, 1)
    τ              ~  HalfCauchy(0, τ₀)
    c²             ~  InvGamma(ν/2, ν s²/2)
    σ              ~  HalfCauchy(0, 1)
    y | β, σ       ~  N(X β, σ²)

Unlike the mean-field VI implementation (HorseshoeRegressor), NUTS samples
the full joint posterior without factorisation assumptions. This gives the
bimodal λ̃ distribution the horseshoe is designed to produce: most edges
are fully shrunk (κ ≈ 1), a few are unshrunk (κ ≈ 0).

Per-gene design
---------------
Each target gene is an independent regression of dimension D (n_tfs).
Fitting all genes jointly would require D × G parameters — intractable for
NUTS at DREAM5 scale (D=99, G=2810 for net2 → 277K parameters).

Instead, one PyMC model per target gene is instantiated sequentially. Each
model has D+D+1+1+1 = 2D+3 continuous parameters (β, λ, τ, c², σ), well
within NUTS capacity (D=99 → 201 parameters per gene).

For full-network AUROC/AUPR: loop over all genes, store posterior-mean β.
For paper figures: store full InferenceData for a showcase subset.
"""
from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az


# ---------------------------------------------------------------------------
# τ₀ from OLS residuals (per-gene, mirrors analytical_horseshoe.py)
# ---------------------------------------------------------------------------

def compute_tau0_from_residuals(
    X: np.ndarray,
    Y: np.ndarray,
    p0: float = 10.0,
    ridge_factor: float = 1e-4,
) -> np.ndarray:
    """Per-gene τ₀ = (p₀/(D−p₀)) · σ_g / √n from OLS residual std.

    Parameters
    ----------
    X : (n, D) design matrix (standardised TF expression)
    Y : (n, G) response matrix (standardised gene expression)
    p0 : expected nonzero TF regulators per gene
    ridge_factor : small ridge for numerical stability

    Returns
    -------
    tau0 : (G,) array of per-gene global scale values
    """
    n, D = X.shape
    G = Y.shape[1]
    XtX = X.T @ X
    reg  = ridge_factor * n * np.eye(D)
    W    = np.linalg.solve(XtX + reg, X.T @ Y)         # (D, G)
    resid = Y - X @ W                                   # (n, G)
    dof   = max(n - D, 1)
    sigma = np.sqrt(np.maximum((resid ** 2).sum(0) / dof, 1e-8))  # (G,)
    tau0  = (p0 / (D - p0)) * sigma / math.sqrt(n)
    return tau0


# ---------------------------------------------------------------------------
# Single-gene PyMC model
# ---------------------------------------------------------------------------

def build_horseshoe_model(
    X: np.ndarray,           # (n, D)
    y: np.ndarray,           # (n,)
    tau0: float,
    nu_slab: float = 4.0,
    s_slab: float  = 2.0,
) -> pm.Model:
    """Return a PyMC model for the regularised horseshoe on one gene."""
    n, D = X.shape
    with pm.Model() as model:
        # Global scale
        tau = pm.HalfCauchy("tau", beta=tau0)

        # Local scales + slab width
        lam = pm.HalfCauchy("lambda", beta=1.0, shape=D)
        c2  = pm.InverseGamma("c2",
                              alpha=nu_slab / 2,
                              beta=nu_slab * s_slab ** 2 / 2)

        # Regularised local scale: λ̃_j = c λ_j / √(c² + τ² λ_j²)
        lam_tilde = pm.Deterministic(
            "lambda_tilde",
            pt.sqrt(c2) * lam / pt.sqrt(c2 + tau ** 2 * lam ** 2),
        )

        # Shrinkage coefficients κ_j = 1 / (1 + λ̃_j²)
        kappa = pm.Deterministic("kappa", 1.0 / (1.0 + lam_tilde ** 2))

        # Regression weights
        beta = pm.Normal("beta", mu=0.0, sigma=lam_tilde * tau, shape=D)

        # Noise
        sigma = pm.HalfCauchy("sigma", beta=1.0)

        # Likelihood
        pm.Normal("y_obs", mu=pt.dot(X, beta), sigma=sigma, observed=y)

    return model


def sample_gene(
    X: np.ndarray,
    y: np.ndarray,
    tau0: float,
    nu_slab: float = 4.0,
    s_slab: float  = 2.0,
    draws: int = 500,
    tune: int  = 500,
    target_accept: float = 0.9,
    chains: int = 2,
    random_seed: int = 42,
    progressbar: bool = False,
) -> az.InferenceData:
    """Run NUTS for a single gene. Returns full InferenceData."""
    model = build_horseshoe_model(X, y, tau0, nu_slab, s_slab)
    with model:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )
    return idata


# ---------------------------------------------------------------------------
# Network-level inference
# ---------------------------------------------------------------------------

def run_pymc_horseshoe(
    network,
    p0: float = 10.0,
    draws: int = 500,
    tune: int  = 500,
    target_accept: float = 0.9,
    chains: int = 2,
    showcase_genes: Sequence[str] | int | None = 20,
    random_seed: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, az.InferenceData]]:
    """Run per-gene NUTS horseshoe for a DREAM5Network.

    Parameters
    ----------
    network :
        DREAM5Network instance.
    p0 :
        Expected number of nonzero TF regulators per gene.
    draws / tune / target_accept / chains :
        PyMC NUTS sampling parameters.
    showcase_genes :
        How many (or which) genes to store full InferenceData for.
        int  → first N genes with highest OLS coefficient magnitude.
        list → specific gene IDs.
        None → no full InferenceData stored.
    random_seed :
        Base seed (incremented per gene for independence).

    Returns
    -------
    scores_df :
        DataFrame [tf, target, score] ranked by posterior |β| mean.
    showcase_idatas :
        Dict mapping gene_id → az.InferenceData for showcase genes.
    """
    import time

    # ----------------------------------------------------------------
    # Prepare standardised expression
    # ----------------------------------------------------------------
    expr  = network.expression.values.astype(np.float64)   # (n, G_all)
    n     = expr.shape[0]
    mean_ = expr.mean(0, keepdims=True)
    std_  = np.maximum(expr.std(0, keepdims=True), 1e-8)
    expr_std = (expr - mean_) / std_

    gene_ids   = network.gene_ids
    tf_ids     = network.tf_ids
    tf_set     = set(tf_ids)
    tf_indices = [i for i, g in enumerate(gene_ids) if g in tf_set]

    X = expr_std[:, tf_indices]   # (n, D)
    Y = expr_std                   # (n, G_all)
    D, G = X.shape[1], Y.shape[1]

    if verbose:
        print(f"  PyMC horseshoe: n={n}  D={D}  G={G}  "
              f"draws={draws}  tune={tune}  chains={chains}")

    # ----------------------------------------------------------------
    # Per-gene tau0 from OLS residuals
    # ----------------------------------------------------------------
    tau0_all = compute_tau0_from_residuals(X, Y, p0=p0)  # (G,)

    # ----------------------------------------------------------------
    # Determine showcase gene indices (by OLS magnitude)
    # ----------------------------------------------------------------
    XtX  = X.T @ X
    W_ols = np.linalg.solve(XtX + 1e-4 * n * np.eye(D), X.T @ Y)  # (D, G)
    max_abs = np.abs(W_ols).max(0)                                  # (G,)

    if showcase_genes is None:
        showcase_indices = set()
        showcase_ids     = []
    elif isinstance(showcase_genes, int):
        showcase_indices = set(int(i) for i in np.argsort(max_abs)[::-1][:showcase_genes])
        showcase_ids     = [gene_ids[i] for i in sorted(showcase_indices)]
    else:
        id_map = {g: i for i, g in enumerate(gene_ids)}
        showcase_indices = set(id_map[g] for g in showcase_genes if g in id_map)
        showcase_ids     = list(showcase_genes)

    # ----------------------------------------------------------------
    # Main loop — per gene
    # ----------------------------------------------------------------
    # posterior_means[d, g] = E[β_d | y_g]
    posterior_means = np.zeros((D, G))
    showcase_idatas: dict[str, az.InferenceData] = {}

    t_start = time.perf_counter()
    n_done  = 0

    for g_idx in range(G):
        gene_id = gene_ids[g_idx]
        # Skip if TF == gene (self-loop; gene is its own TF)
        y_g    = Y[:, g_idx]
        tau0_g = float(tau0_all[g_idx])
        seed_g = random_seed + g_idx

        store_full = g_idx in showcase_indices

        try:
            idata = sample_gene(
                X, y_g, tau0_g,
                draws=draws, tune=tune,
                target_accept=target_accept,
                chains=chains,
                random_seed=seed_g,
                progressbar=False,
            )
            beta_post = idata.posterior["beta"].values   # (chains, draws, D)
            posterior_means[:, g_idx] = np.abs(beta_post).mean((0, 1))

            if store_full:
                showcase_idatas[gene_id] = idata

        except Exception as e:
            if verbose:
                print(f"    gene {gene_id} ({g_idx}/{G}) failed: {e}")
            # Fallback to OLS
            posterior_means[:, g_idx] = np.abs(W_ols[:, g_idx])

        n_done += 1
        if verbose and n_done % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate    = n_done / elapsed
            eta     = (G - n_done) / rate
            print(f"    {n_done}/{G} genes  ({elapsed:.0f}s elapsed, "
                  f"ETA {eta/60:.1f} min)")

    elapsed_total = time.perf_counter() - t_start
    if verbose:
        print(f"  Completed {G} genes in {elapsed_total/60:.1f} min")

    # ----------------------------------------------------------------
    # Build ranked edge DataFrame
    # ----------------------------------------------------------------
    tf_arr   = np.array(tf_ids)
    gene_arr = np.array(gene_ids)
    tf_names   = np.repeat(tf_arr[:, None],  G, axis=1)   # (D, G)
    gene_names = np.repeat(gene_arr[None, :], D, axis=0)  # (D, G)
    not_self   = tf_names != gene_names

    scores_df = pd.DataFrame({
        "tf":     tf_names[not_self],
        "target": gene_names[not_self],
        "score":  posterior_means[not_self],
    })
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)

    return scores_df, showcase_idatas
