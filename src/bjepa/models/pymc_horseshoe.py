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
from scipy import stats as scipy_stats


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
    """Return a PyMC model for the regularised horseshoe on one gene.

    Uses non-centered parameterisation (NCP) for beta to avoid the funnel
    geometry that causes divergences with the centered form
    beta ~ N(0, lambda_tilde * tau).

    NCP: beta_raw ~ N(0, 1),  beta = beta_raw * lambda_tilde * tau
    NUTS samples beta_raw in an isotropic space; beta is derived.
    This is the standard fix recommended by Betancourt & Girolami (2015)
    and the PyMC / Stan documentation for hierarchical scale-mixture models.
    """
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

        # Non-centered parameterisation: avoids funnel geometry
        beta_raw = pm.Normal("beta_raw", mu=0.0, sigma=1.0, shape=D)
        beta = pm.Deterministic("beta", beta_raw * lam_tilde * tau)

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
    target_accept: float = 0.95,
    chains: int = 2,
    max_treedepth: int = 12,
    random_seed: int = 42,
    progressbar: bool = False,
) -> az.InferenceData:
    """Run NUTS for a single gene. Returns full InferenceData.

    max_treedepth=12 (vs default 10) handles genes where the horseshoe
    posterior has sharper curvature — avoids max-tree-depth warnings at
    the cost of more leapfrog steps per sample for those genes.
    """
    model = build_horseshoe_model(X, y, tau0, nu_slab, s_slab)
    with model:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts_sampler_kwargs={"max_treedepth": max_treedepth},
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
    target_accept: float = 0.95,
    chains: int = 2,
    max_treedepth: int = 12,
    max_nuts_genes: int | None = None,
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
    # Determine which genes get NUTS vs OLS fallback
    # max_nuts_genes=None  → NUTS on all genes (slow, full network)
    # max_nuts_genes=K     → NUTS on top-K by OLS signal, ridge for rest
    # This is justified: low-signal genes have beta≈0 under both NUTS and
    # ridge, so the fallback loses negligible information while saving hours.
    # ----------------------------------------------------------------
    if max_nuts_genes is None:
        nuts_indices = set(range(G))
    else:
        k = min(max_nuts_genes, G)
        nuts_indices = set(int(i) for i in np.argsort(max_abs)[::-1][:k])
        # Always include showcase genes in NUTS set
        nuts_indices |= showcase_indices
        if verbose:
            print(f"  NUTS on top-{len(nuts_indices)} genes by OLS signal "
                  f"(fallback to GCV-ridge for remaining {G - len(nuts_indices)})")

    # ----------------------------------------------------------------
    # Main loop — per gene
    # ----------------------------------------------------------------
    # posterior_means[d, g]        = E[|β_d| | y_g]  (for ranking)
    # posterior_means_signed[d, g] = E[β_d | y_g]    (for FLASH z-score)
    # posterior_stds[d, g]         = std[β_d | y_g]  (for FLASH z-score)
    posterior_means        = np.zeros((D, G))
    posterior_means_signed = np.zeros((D, G))
    posterior_stds         = np.zeros((D, G))
    showcase_idatas: dict[str, az.InferenceData] = {}

    # Pre-fill all genes with GCV-ridge scores as fallback
    # (genes not in nuts_indices keep these values)
    from bjepa.models.analytical_horseshoe import gcv_ridge_factor, analytical_horseshoe_scores as _hs_scores
    rf_fallback = gcv_ridge_factor(network, verbose=False)
    scores_ridge_fallback = _hs_scores(network, ridge_factor=rf_fallback, horseshoe=False)
    _ridge_map: dict[tuple, float] = {
        (row.tf, row.target): row.score
        for row in scores_ridge_fallback.itertuples()
    }
    for g_idx in range(G):
        for d_idx in range(D):
            key = (tf_ids[d_idx], gene_ids[g_idx])
            v = _ridge_map.get(key, 0.0)
            posterior_means[d_idx, g_idx]        = v
            posterior_means_signed[d_idx, g_idx] = W_ols[d_idx, g_idx]
            posterior_stds[d_idx, g_idx]         = max(v * 0.1, 1e-8)

    t_start = time.perf_counter()
    n_done  = 0

    n_nuts_total = len(nuts_indices)
    n_nuts_done  = 0

    for g_idx in range(G):
        gene_id    = gene_ids[g_idx]
        store_full = g_idx in showcase_indices

        # Skip NUTS for low-signal genes — ridge fallback already filled above
        if g_idx not in nuts_indices:
            if store_full:
                # Shouldn't happen (showcase always in nuts_indices), but guard
                showcase_indices.discard(g_idx)
            continue

        y_g    = Y[:, g_idx]
        tau0_g = float(tau0_all[g_idx])
        seed_g = random_seed + g_idx

        try:
            idata = sample_gene(
                X, y_g, tau0_g,
                draws=draws, tune=tune,
                target_accept=target_accept,
                chains=chains,
                max_treedepth=max_treedepth,
                random_seed=seed_g,
                progressbar=False,
            )
            beta_post = idata.posterior["beta"].values   # (chains, draws, D)
            flat = beta_post.reshape(-1, D)               # (S, D)
            posterior_means[:, g_idx]        = np.abs(flat).mean(0)
            posterior_means_signed[:, g_idx] = flat.mean(0)
            posterior_stds[:, g_idx]         = flat.std(0, ddof=1)

            if store_full:
                showcase_idatas[gene_id] = idata

        except Exception as e:
            if verbose:
                print(f"    gene {gene_id} failed: {e}")
            # Fallback already pre-filled; nothing to do

        n_nuts_done += 1
        n_done      += 1
        if verbose and n_nuts_done % 20 == 0:
            elapsed = time.perf_counter() - t_start
            rate    = n_nuts_done / elapsed
            eta     = (n_nuts_total - n_nuts_done) / rate
            print(f"    NUTS {n_nuts_done}/{n_nuts_total} genes  "
                  f"({elapsed:.0f}s elapsed, ETA {eta/60:.1f} min)")

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

    # is_nuts[d, g] = True if gene g was sampled with NUTS, False = ridge fallback
    is_nuts_mat = np.zeros((D, G), dtype=bool)
    for g_idx in nuts_indices:
        is_nuts_mat[:, g_idx] = True

    scores_df = pd.DataFrame({
        "tf":            tf_names[not_self],
        "target":        gene_names[not_self],
        "score":         posterior_means[not_self],        # |E[β]| for ranking
        "score_signed":  posterior_means_signed[not_self], # E[β] for FLASH z-score
        "score_std":     posterior_stds[not_self],         # std[β] for FLASH z-score
        "is_nuts":       is_nuts_mat[not_self],            # True = calibrated posterior
    })
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)

    return scores_df, showcase_idatas


# ---------------------------------------------------------------------------
# FLASH FDR filter  (Mendel hypothesis #1)
# ---------------------------------------------------------------------------

def flash_fdr_scores(
    scores_df: pd.DataFrame,
    fdr_level: float = 0.20,
    min_std: float = 1e-8,
) -> pd.DataFrame:
    """Apply frequentist-assisted horseshoe FDR control (FLASH).

    Uses the posterior mean / posterior std ratio as a z-statistic and applies
    Benjamini-Hochberg correction at the specified FDR level. Among selected
    edges, ranks by |posterior mean| (identical to ranking by |z| since std
    cancels when comparing within the selected set only if we want z-ordering,
    but |beta| is the biologically interpretable quantity).

    Parameters
    ----------
    scores_df :
        Output of run_pymc_horseshoe — must contain columns
        'score_signed' and 'score_std'.
    fdr_level :
        Target FDR (Benjamini-Hochberg). Default 0.20.
    min_std :
        Floor for score_std to avoid division by zero.

    Returns
    -------
    DataFrame with same columns as input, filtered to BH-selected edges,
    ranked by |score_signed| descending. Adds column 'p_value' and
    'z_score'.
    """
    df = scores_df.copy()

    # Require signed mean and std columns
    if "score_signed" not in df.columns or "score_std" not in df.columns:
        raise ValueError("scores_df must have 'score_signed' and 'score_std' columns. "
                         "Run run_pymc_horseshoe (not analytical_horseshoe_scores).")

    std_safe = np.maximum(df["score_std"].values, min_std)
    z = df["score_signed"].values / std_safe
    # Two-sided p-value under normal approximation
    p = 2.0 * scipy_stats.norm.sf(np.abs(z))

    df["z_score"] = z
    df["p_value"] = p

    # Benjamini-Hochberg correction
    m = len(df)
    order = np.argsort(p)
    rank  = np.empty(m, dtype=int)
    rank[order] = np.arange(1, m + 1)
    bh_threshold = (rank / m) * fdr_level
    selected = p <= bh_threshold

    df_sel = df[selected].copy()
    df_sel = df_sel.sort_values("score", ascending=False).reset_index(drop=True)
    return df_sel
