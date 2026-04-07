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


def gcv_ridge_factor(
    network,
    alpha_grid: np.ndarray | None = None,
    verbose: bool = True,
) -> float:
    """Find the ridge_factor minimising GCV prediction error via SVD.

    GCV(α) = ||Y - X·Ŵ_α||²_F / (n · (1 - trace(H_α)/n)²)

    where H_α = X(XᵀX + αI)⁻¹Xᵀ and trace(H_α) = Σⱼ dⱼ²/(dⱼ² + α),
    with dⱼ the singular values of X.  Evaluating GCV across a grid of α
    requires only one SVD of X (O(n·p²)) plus O(|grid|·n·p) work for
    residuals — all negligible compared to expression data loading.

    Parameters
    ----------
    network:
        DREAM5Network instance.
    alpha_grid:
        Absolute ridge values to search (not ridge_factor).
        Defaults to 60 log-spaced values from 1e-4 to 1e4·n.
    verbose:
        Print GCV curve and selected α.

    Returns
    -------
    float
        Optimal ridge_factor = α* / n.
    """
    expr = network.expression.values.astype(np.float64)
    n    = expr.shape[0]

    gene_mean = expr.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr.std(axis=0, keepdims=True), 1e-8)
    expr_std  = (expr - gene_mean) / gene_std

    gene_ids   = network.gene_ids
    tf_set     = set(network.tf_ids)
    tf_indices = [i for i, g in enumerate(gene_ids) if g in tf_set]

    X = expr_std[:, tf_indices]   # (n, p)
    Y = expr_std                   # (n, G)
    p = X.shape[1]

    # SVD of X once
    U, d, Vt = np.linalg.svd(X, full_matrices=False)   # d: (p,)
    d2 = d ** 2                                          # (p,)
    UtY = U.T @ Y                                        # (p, G)
    residual_base = Y - U @ UtY                          # component of Y orthogonal to col(X)

    if alpha_grid is None:
        alpha_grid = np.logspace(-4, math.log10(1e4 * n), 60)

    gcv_scores = []
    for alpha in alpha_grid:
        # Hat-matrix trace: tr(H_α) = Σⱼ dⱼ²/(dⱼ² + α)
        trace_H = (d2 / (d2 + alpha)).sum()

        # Fitted values: Ŷ = U · diag(dⱼ²/(dⱼ²+α)) · UᵀY
        shrink   = d2 / (d2 + alpha)                   # (p,)
        Y_hat    = U @ (shrink[:, np.newaxis] * UtY)   # (n, G)
        residuals = Y - Y_hat                           # (n, G)

        rss = (residuals ** 2).sum()                   # scalar
        denom = n * (1.0 - trace_H / n) ** 2
        gcv_scores.append(rss / denom)

    gcv_scores = np.array(gcv_scores)
    best_idx   = int(np.argmin(gcv_scores))
    alpha_star = alpha_grid[best_idx]
    rf_star    = alpha_star / n

    if verbose:
        print(f"  GCV calibration  n={n}  p={p}  n/p={n/p:.2f}")
        print(f"  alpha* = {alpha_star:.4g}  →  ridge_factor = {rf_star:.4g}  "
              f"(GCV={gcv_scores[best_idx]:.4g})")
        # Show a few landmark GCV values
        landmarks = [0, len(alpha_grid)//4, len(alpha_grid)//2,
                     3*len(alpha_grid)//4, len(alpha_grid)-1]
        for i in landmarks:
            marker = " ← best" if i == best_idx else ""
            print(f"    α={alpha_grid[i]:.3g}  rf={alpha_grid[i]/n:.3g}  "
                  f"GCV={gcv_scores[i]:.4g}{marker}")

    return rf_star
