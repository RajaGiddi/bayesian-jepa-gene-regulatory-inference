"""Analytical (closed-form) horseshoe for GRN edge scoring.

Two modes
---------
OLS mode (W_prior_mean=None):
    1. OLS estimate:   β̂_g = (XᵀX + δI)⁻¹ Xᵀ y_g
    2. Residual var:   σ²_g = ||y_g - X β̂_g||² / (n - p)
    3. Per-gene τ₀:    τ₀_g = (p₀ / (D - p₀)) · σ_g / √n
    4. Shrinkage:      κ_{i,g} = 1 / (1 + n·β̂_{i,g}² / (σ²_g · τ₀_g²))
    5. HS estimate:    β_{hs,i,g} = (1 - κ_{i,g}) · β̂_{i,g}

MAP mode (W_prior_mean = W_init from Stage 1):
    Replaces the OLS step with a Gaussian MAP estimate that uses W_init
    as the prior mean with precision α (Tikhonov regularization):

        W_map = solve(XᵀX + α·I,  XᵀY + α·W_init)

    This is the posterior mode under:
        β | W_init  ~  N(W_init, (1/α)·I)   (prior from JEPA Stage 1)
        y | β, X    ~  N(X·β, σ²·I)          (likelihood)

    The JEPA warm start provides a structured prior mean that (a) stabilises
    ill-conditioned networks (net2: n/p = 1.6) and (b) biases the MAP
    solution toward the JEPA co-expression signal before the horseshoe
    applies selective shrinkage.

All genes are solved simultaneously via one batched LAPACK call.

Why this works where mean-field VI fails
-----------------------------------------
Mean-field VI for the horseshoe has a degenerate fixed point: the KL
gradient on m_lambda is ~500× larger than the regression gradient, so
λ̃ → 1 for all edges (κ ≈ 0.49, no sparsity differentiation). The
closed-form solution bypasses VI entirely — shrinkage is computed
exactly from the posterior mode given the OLS/MAP likelihood.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def analytical_horseshoe_scores(
    network,
    p0: float = 10.0,
    ridge_factor: float = 1e-6,
    W_prior_mean: np.ndarray | None = None,
    alpha: float | None = None,
    horseshoe: bool = True,
) -> pd.DataFrame:
    """Compute analytical horseshoe edge scores for a DREAM5Network.

    Parameters
    ----------
    network:
        A DREAM5Network instance with .expression, .gene_ids, .tf_ids.
    p0:
        Expected number of nonzero TF regulators per gene.
        Typical: 10–20.
    ridge_factor:
        Numerical stability ridge when W_prior_mean is None.
        δ = ridge_factor · n. Ignored when alpha is set.
    W_prior_mean:
        Optional MAP prior mean for W, shape (n_tfs, n_genes).
        Pass W_init from Stage 1 to activate MAP mode.
        If None, falls back to pure OLS with numerical ridge.
    alpha:
        MAP prior precision (Tikhonov strength).
        W_map = solve(XᵀX + α·I, XᵀY + α·W_init).
        Defaults to n (one sample worth of prior weight) when
        W_prior_mean is provided and alpha is None.
        Larger α → solution pulled more toward W_prior_mean.
        Typical useful range: 0.1·n to 10·n.

    Returns
    -------
    pd.DataFrame
        Columns: [tf, target, score] sorted descending by |beta_hs|.
        Excludes self-loops (tf == target).
    """
    expr = network.expression.values.astype(np.float64)  # (n_samples, n_genes)
    n, n_all_genes = expr.shape

    # Per-gene standardization (same as trainer)
    gene_mean = expr.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr.std(axis=0, keepdims=True), 1e-8)
    expr_std  = (expr - gene_mean) / gene_std              # (n_samples, n_genes)

    # TF indices in the gene list
    gene_ids  = network.gene_ids
    tf_ids    = network.tf_ids
    tf_set    = set(tf_ids)
    tf_indices = [i for i, g in enumerate(gene_ids) if g in tf_set]

    X = expr_std[:, tf_indices]  # (n_samples, n_tfs)
    Y = expr_std                  # (n_samples, n_genes)

    n_tfs  = X.shape[1]
    n_genes = Y.shape[1]
    D = n_tfs

    # ----------------------------------------------------------------
    # Step 1: MAP or OLS — solve for W given all genes at once
    # ----------------------------------------------------------------
    XtX = X.T @ X                              # (n_tfs, n_tfs)
    XtY = X.T @ Y                              # (n_tfs, n_genes)

    if W_prior_mean is not None:
        _alpha = alpha if alpha is not None else float(n)
        reg    = _alpha * np.eye(n_tfs)
        rhs    = XtY + _alpha * W_prior_mean   # (n_tfs, n_genes)
        W_ols  = np.linalg.solve(XtX + reg, rhs)
    else:
        delta  = ridge_factor * n * np.eye(n_tfs)
        W_ols  = np.linalg.solve(XtX + delta, XtY)

    # ----------------------------------------------------------------
    # Step 2: Per-gene residual variance
    # ----------------------------------------------------------------
    Y_hat    = X @ W_ols                       # (n_samples, n_genes)
    residuals = Y - Y_hat                      # (n_samples, n_genes)
    dof      = max(n - n_tfs, 1)
    sigma2   = (residuals ** 2).sum(axis=0) / dof  # (n_genes,)
    sigma2   = np.maximum(sigma2, 1e-8)

    # ----------------------------------------------------------------
    # Step 3–5: Horseshoe shrinkage (skipped if horseshoe=False)
    # ----------------------------------------------------------------
    if horseshoe:
        if D <= p0:
            raise ValueError(
                f"p0={p0} >= n_tfs={D}: too many expected nonzeros. Reduce p0."
            )
        sigma_j  = np.sqrt(sigma2)                          # (n_genes,)
        tau0_j   = (p0 / (D - p0)) * sigma_j / math.sqrt(n)  # (n_genes,)
        beta2    = W_ols ** 2                               # (n_tfs, n_genes)
        prior_var = sigma2[np.newaxis, :] * tau0_j[np.newaxis, :] ** 2
        kappa    = 1.0 / (1.0 + n * beta2 / np.maximum(prior_var, 1e-20))
        abs_beta = np.abs((1.0 - kappa) * W_ols)
    else:
        # Ridge-only ablation: skip shrinkage, rank by |W_ols| directly
        abs_beta = np.abs(W_ols)

    # ----------------------------------------------------------------
    # Step 6: Build edge DataFrame — vectorised, exclude self-loops
    # ----------------------------------------------------------------
    tf_arr   = np.array(tf_ids)               # (n_tfs,)
    gene_arr = np.array(gene_ids)             # (n_genes,)

    # Broadcast names to (n_tfs, n_genes)
    tf_names   = np.repeat(tf_arr[:, np.newaxis],  n_genes, axis=1)
    gene_names = np.repeat(gene_arr[np.newaxis, :], n_tfs, axis=0)
    not_self   = tf_names != gene_names

    df = pd.DataFrame({
        "tf":     tf_names[not_self],
        "target": gene_names[not_self],
        "score":  abs_beta[not_self],
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)
