"""Phase 3: Horseshoe-prior MAP for GRN structure + dynamics.

Two models are provided:

Phase 3A (superseded): logit_G * w_full — non-identifiable product.

Phase 3B — Regularised horseshoe on W (Piironen & Vehtari 2017):

    tau        ~ HalfNormal(tau0)         [global shrinkage]
    lambda_ij  ~ HalfCauchy(1)            [local scales, off-diagonal only]
    c2         ~ InverseGamma(2, 8)       [slab regularisation]
    lambda_tilde = sqrt(c2*lambda^2 / (c2 + tau^2*lambda^2))
    w_ij       ~ Normal(0, tau * lambda_tilde_ij)

Edge score = |W_ij|.  No separate G — sparsity IS the graph.

Why this beats GENIE3:
  - Exact do-calculus knockout likelihood (causal, not correlational)
  - Horseshoe identifies which edges are needed to explain ODE dynamics
  - Gradient flows directly through W_ij (identifiable, unlike W*G_soft)

Usage:
    from grn_world_model.phase3_joint import run_phase3_horseshoe, compute_aupr

    result = run_phase3_horseshoe(net, num_steps=3000)
    aupr   = compute_aupr(result['edge_scores'], net.gold_standard)
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from grn_world_model.data_loader import DREAM4Network
from grn_world_model.jax_model import _solve_timeseries, _knockout_steady_state
from grn_world_model.ode_model import apply_knockout

# logit(0.1) ≈ -2.197 — prior edge probability ~10%
_SPARSITY_LOGIT = float(np.log(0.1 / 0.9))


# ---------------------------------------------------------------------------
# Joint numpyro model
# ---------------------------------------------------------------------------

def grn_model_joint(
    net: DREAM4Network,
    use_knockouts: bool = True,
    use_timeseries: bool = True,
    w_prior_sigma: float = 0.5,
    gamma_prior_sigma: float = 1.0,
    sigma_obs_prior: float = 0.5,
    sparsity_logit: float = _SPARSITY_LOGIT,
    stability_coef: float = 5.0,
) -> None:
    """Numpyro model for joint graph + dynamics inference.

    Latent variables
    ----------------
    logit_G : (N*N,)  — log-odds of each directed edge (diagonal fixed to -10)
    w_full  : (N, N)  — edge weight for every potential interaction
    gamma   : (N,)    — degradation rates (HalfNormal)
    sigma_obs: (N,)   — observation noise (HalfNormal)

    Effective interaction matrix: W_masked = w_full * sigmoid(logit_G)
    """
    N = net.n_genes
    wildtype = jnp.array(net.wildtype, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Graph prior
    # ------------------------------------------------------------------
    logit_G_flat = numpyro.sample(
        'logit_G',
        dist.Normal(sparsity_logit * jnp.ones(N * N), jnp.ones(N * N)),
    )
    logit_G = logit_G_flat.reshape(N, N)
    # Force diagonal to -10 (no self-regulation)
    diag_mask = jnp.eye(N, dtype=jnp.float32)
    logit_G   = logit_G * (1 - diag_mask) - 10.0 * diag_mask
    G_soft    = jax.nn.sigmoid(logit_G)     # (N, N) ∈ [0, 1]

    # ------------------------------------------------------------------
    # Dynamics priors
    # ------------------------------------------------------------------
    w_full = numpyro.sample(
        'w_full',
        dist.Normal(jnp.zeros((N, N)), w_prior_sigma * jnp.ones((N, N))),
    )
    gamma = numpyro.sample(
        'gamma',
        dist.HalfNormal(gamma_prior_sigma * jnp.ones(N)),
    )
    sigma_obs = numpyro.sample(
        'sigma_obs',
        dist.HalfNormal(sigma_obs_prior * jnp.ones(N)),
    )

    # ------------------------------------------------------------------
    # Soft-masked interaction matrix
    # ------------------------------------------------------------------
    W_masked = w_full * G_soft              # (N, N)

    # Gershgorin stability penalty
    if stability_coef > 0:
        row_excess = jnp.sum(jnp.abs(W_masked), axis=1) - gamma
        numpyro.factor(
            'stability',
            -stability_coef * jnp.sum(jnp.maximum(row_excess, 0.0)),
        )

    # Basal rates: b_i = gamma_i * wt_i - (W_masked @ wt)_i
    b = gamma * wildtype - W_masked @ wildtype

    # ------------------------------------------------------------------
    # Timeseries likelihood (ODE)
    # ------------------------------------------------------------------
    if use_timeseries:
        for i, ts in enumerate(net.timeseries):
            t0_f        = float(ts.timepoints[0])
            t1_f        = float(ts.timepoints[-1])
            save_ts     = jnp.array(ts.timepoints[1:], dtype=jnp.float32)
            x0          = jnp.array(ts.expression[0],  dtype=jnp.float32)
            perturbation = jnp.array(ts.perturbation,   dtype=jnp.float32)
            observed    = jnp.array(ts.expression[1:],  dtype=jnp.float32)

            x_pred = _solve_timeseries(
                W_masked, gamma, b, x0, t0_f, t1_f, save_ts, perturbation,
            )
            numpyro.sample(
                f'obs_ts_{i}',
                dist.Normal(x_pred, sigma_obs[None, :]),
                obs=observed,
            )

    # ------------------------------------------------------------------
    # Knockout likelihood (analytical; differentiable through G_soft)
    # ------------------------------------------------------------------
    if use_knockouts:
        for gene_idx, ko_expression in net.knockout_pairs:
            # Zero out row + col for knocked-out gene in soft graph
            G_ko_soft = G_soft \
                .at[:, gene_idx].set(0.0) \
                .at[gene_idx, :].set(0.0)
            x_ss = _knockout_steady_state(
                w_full, G_ko_soft, gamma, wildtype, gene_idx,
            )
            numpyro.sample(
                f'obs_ko_{gene_idx}',
                dist.Normal(x_ss, sigma_obs),
                obs=jnp.array(ko_expression, dtype=jnp.float32),
            )


# ---------------------------------------------------------------------------
# MAP optimisation
# ---------------------------------------------------------------------------

def run_phase3(
    net: DREAM4Network,
    use_knockouts: bool = True,
    use_timeseries: bool = True,
    num_steps: int = 3000,
    lr: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
    **model_kwargs,
) -> dict:
    """Run joint MAP for graph + dynamics. Returns result dict.

    Keys in result
    --------------
    edge_probs  : (N, N) sigmoid(logit_G) — edge confidence, use for AUPR
    w_full      : (N, N) MAP edge weights
    gamma       : (N,)   MAP degradation rates
    sigma_obs   : (N,)   MAP noise scales
    losses      : (num_steps,) SVI ELBO loss curve
    """
    def model():
        grn_model_joint(net, use_knockouts=use_knockouts,
                        use_timeseries=use_timeseries, **model_kwargs)

    guide     = AutoDelta(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=1.0)
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

    if verbose:
        print(f'Phase 3 MAP: {num_steps} steps, lr={lr}', flush=True)

    result = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=verbose)

    params = guide.median(result.params)

    N = net.n_genes
    logit_G = np.array(params['logit_G']).reshape(N, N)
    # Re-apply diagonal masking
    np.fill_diagonal(logit_G, -10.0)

    return {
        'edge_probs': 1.0 / (1.0 + np.exp(-logit_G)),   # sigmoid
        'logit_G':    logit_G,
        'w_full':     np.array(params['w_full']),
        'gamma':      np.array(params['gamma']),
        'sigma_obs':  np.array(params['sigma_obs']),
        'losses':     np.array(result.losses),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_aupr(edge_probs: np.ndarray, gold_standard: np.ndarray) -> float:
    """Compute AUPR using edge_probs (N×N) as scores vs binary gold_standard.

    Excludes diagonal (no self-loops in gold standard).
    """
    from sklearn.metrics import average_precision_score
    N = edge_probs.shape[0]
    mask = ~np.eye(N, dtype=bool)
    scores = edge_probs[mask]
    labels = gold_standard[mask].astype(int)
    return float(average_precision_score(labels, scores))


def compute_auroc(edge_probs: np.ndarray, gold_standard: np.ndarray) -> float:
    """Compute AUROC using edge_probs (N×N) as scores vs binary gold_standard."""
    from sklearn.metrics import roc_auc_score
    N = edge_probs.shape[0]
    mask = ~np.eye(N, dtype=bool)
    scores = edge_probs[mask]
    labels = gold_standard[mask].astype(int)
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Phase 3B: Regularised horseshoe model
# ---------------------------------------------------------------------------

def grn_model_horseshoe(
    net: DREAM4Network,
    use_knockouts: bool = True,
    use_timeseries: bool = True,
    gamma_prior_sigma: float = 1.0,
    sigma_obs_prior: float = 0.5,
    stability_coef: float = 5.0,
    tau0: float = 0.05,
) -> None:
    """Numpyro model with regularised horseshoe prior on edge weights.

    Graph structure is implicit in W: edge (i,j) exists iff |W_ij| is large.
    No separate G variable — eliminates the W*G non-identifiability.

    Horseshoe parameterisation (Piironen & Vehtari 2017):
        tau        ~ HalfNormal(tau0)
        lambda_ij  ~ HalfCauchy(1)
        c2         ~ InverseGamma(2, 8)
        lambda_tilde_ij = sqrt(c2 * lambda^2 / (c2 + tau^2 * lambda^2))
        w_ij       ~ Normal(0, tau * lambda_tilde_ij)

    tau0 controls global sparsity — smaller = more shrinkage toward 0.
    """
    N   = net.n_genes
    D   = N * (N - 1)                           # off-diagonal edges
    wildtype = jnp.array(net.wildtype, dtype=jnp.float32)

    # Precompute off-diagonal index arrays (fixed, numpy)
    rows, cols = np.where(~np.eye(N, dtype=bool))  # (D,), (D,)

    # ------------------------------------------------------------------
    # Regularised horseshoe prior on off-diagonal weights
    # ------------------------------------------------------------------
    tau = numpyro.sample('tau', dist.HalfNormal(tau0))

    lambda_ij = numpyro.sample('lambda_ij', dist.HalfCauchy(jnp.ones(D)))
    c2        = numpyro.sample('c2', dist.InverseGamma(2.0, 8.0))

    # Regularised local scales: caps extreme lambda values
    lambda_tilde = jnp.sqrt(
        c2 * lambda_ij ** 2 / (c2 + tau ** 2 * lambda_ij ** 2)
    )

    w_raw = numpyro.sample('w_raw', dist.Normal(jnp.zeros(D), jnp.ones(D)))
    w_od  = w_raw * tau * lambda_tilde           # off-diagonal weights (D,)

    # Build full N×N weight matrix (diagonal = 0)
    W = jnp.zeros((N, N)).at[rows, cols].set(w_od)

    # ------------------------------------------------------------------
    # Dynamics priors
    # ------------------------------------------------------------------
    gamma     = numpyro.sample('gamma',     dist.HalfNormal(gamma_prior_sigma * jnp.ones(N)))
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(sigma_obs_prior   * jnp.ones(N)))

    # ------------------------------------------------------------------
    # Stability penalty (Gershgorin)
    # ------------------------------------------------------------------
    if stability_coef > 0:
        row_excess = jnp.sum(jnp.abs(W), axis=1) - gamma
        numpyro.factor('stability',
                       -stability_coef * jnp.sum(jnp.maximum(row_excess, 0.0)))

    # Basal rates
    b = gamma * wildtype - W @ wildtype

    # ------------------------------------------------------------------
    # Timeseries likelihood
    # ------------------------------------------------------------------
    if use_timeseries:
        for i, ts in enumerate(net.timeseries):
            t0_f        = float(ts.timepoints[0])
            t1_f        = float(ts.timepoints[-1])
            save_ts     = jnp.array(ts.timepoints[1:], dtype=jnp.float32)
            x0          = jnp.array(ts.expression[0],  dtype=jnp.float32)
            perturbation = jnp.array(ts.perturbation,   dtype=jnp.float32)
            observed    = jnp.array(ts.expression[1:],  dtype=jnp.float32)
            x_pred = _solve_timeseries(W, gamma, b, x0, t0_f, t1_f,
                                       save_ts, perturbation)
            numpyro.sample(f'obs_ts_{i}',
                           dist.Normal(x_pred, sigma_obs[None, :]),
                           obs=observed)

    # ------------------------------------------------------------------
    # Knockout likelihood (differentiable: zero row+col in W)
    # ------------------------------------------------------------------
    if use_knockouts:
        for gene_idx, ko_expression in net.knockout_pairs:
            # Zero out interactions involving knocked-out gene
            W_ko = W.at[:, gene_idx].set(0.0).at[gene_idx, :].set(0.0)
            b_ko = gamma * wildtype - W_ko @ wildtype
            b_ko = b_ko.at[gene_idx].set(0.0)
            A    = jnp.diag(gamma) - W_ko
            x_ss = jnp.linalg.solve(A, b_ko).at[gene_idx].set(0.0)
            numpyro.sample(f'obs_ko_{gene_idx}',
                           dist.Normal(x_ss, sigma_obs),
                           obs=jnp.array(ko_expression, dtype=jnp.float32))


def run_phase3_horseshoe(
    net: DREAM4Network,
    use_knockouts: bool = True,
    use_timeseries: bool = True,
    num_steps: int = 3000,
    lr: float = 0.01,
    tau0: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
    **model_kwargs,
) -> dict:
    """Run regularised horseshoe MAP. Returns result dict.

    Keys
    ----
    edge_scores : (N, N)  |W_ij| — use for AUPR (diagonal = 0)
    W           : (N, N)  MAP edge weights
    gamma       : (N,)    MAP degradation rates
    sigma_obs   : (N,)    MAP noise scales
    losses      : (num_steps,) SVI loss curve
    """
    def model():
        grn_model_horseshoe(net, use_knockouts=use_knockouts,
                            use_timeseries=use_timeseries,
                            tau0=tau0, **model_kwargs)

    guide     = AutoDelta(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=1.0)
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

    if verbose:
        print(f'Phase 3 horseshoe MAP: {num_steps} steps, lr={lr}, tau0={tau0}',
              flush=True)

    result = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=verbose)
    params = guide.median(result.params)

    N    = net.n_genes
    rows, cols = np.where(~np.eye(N, dtype=bool))
    w_od = np.array(params['w_raw']) * float(params['tau']) * np.array(
        np.sqrt(float(params['c2']) * params['lambda_ij'] ** 2 /
                (float(params['c2']) + float(params['tau']) ** 2 * params['lambda_ij'] ** 2))
    )
    W = np.zeros((N, N))
    W[rows, cols] = w_od

    edge_scores = np.abs(W)
    np.fill_diagonal(edge_scores, 0.0)

    return {
        'edge_scores': edge_scores,
        'W':          W,
        'gamma':      np.array(params['gamma']),
        'sigma_obs':  np.array(params['sigma_obs']),
        'losses':     np.array(result.losses),
    }


# ---------------------------------------------------------------------------
# Phase 3C: Gradient matching + horseshoe (recommended)
# ---------------------------------------------------------------------------

def _compute_gradients(net: DREAM4Network) -> list[np.ndarray]:
    """Finite-difference dx/dt for each timeseries. Returns list of (T, N)."""
    grads = []
    for ts in net.timeseries:
        # np.gradient uses central differences at interior, one-sided at edges
        dx_dt = np.gradient(ts.expression, ts.timepoints, axis=0)  # (T, N)
        grads.append(dx_dt.astype(np.float32))
    return grads


def grn_model_gradient_matching(
    net: DREAM4Network,
    dx_dt_list: list,           # precomputed, list of (T, N) arrays
    use_knockouts: bool = True,
    gamma_prior_sigma: float = 1.0,
    sigma_obs_prior: float = 0.5,
    tau0: float = 0.1,
) -> None:
    """Gradient-matching + regularised horseshoe for GRN inference.

    Linearises the ODE:
        dx_i/dt = b_i + Σ_j W_ij·x_j(t) − γ_i·x_i(t)

    This is linear in W given observed (x(t), dx/dt(t)) — sparse linear
    regression that the horseshoe was designed for.  Combined with exact
    knockout steady-state constraints this should identify edges well.

    No ODE integration — dx/dt estimated by finite differences.
    """
    N        = net.n_genes
    wildtype = jnp.array(net.wildtype, dtype=jnp.float32)
    rows, cols = np.where(~np.eye(N, dtype=bool))   # off-diagonal indices
    D = len(rows)

    # ------------------------------------------------------------------
    # Regularised horseshoe on off-diagonal weights
    # ------------------------------------------------------------------
    tau       = numpyro.sample('tau',       dist.HalfNormal(tau0))
    lambda_ij = numpyro.sample('lambda_ij', dist.HalfCauchy(jnp.ones(D)))
    c2        = numpyro.sample('c2',        dist.InverseGamma(2.0, 8.0))
    lambda_tilde = jnp.sqrt(c2 * lambda_ij**2 / (c2 + tau**2 * lambda_ij**2))
    w_raw = numpyro.sample('w_raw', dist.Normal(jnp.zeros(D), jnp.ones(D)))
    w_od  = w_raw * tau * lambda_tilde
    W     = jnp.zeros((N, N)).at[rows, cols].set(w_od)

    gamma     = numpyro.sample('gamma',     dist.HalfNormal(gamma_prior_sigma * jnp.ones(N)))
    sigma_gm  = numpyro.sample('sigma_gm',  dist.HalfNormal(sigma_obs_prior   * jnp.ones(N)))

    # Basal rates from wildtype steady-state constraint
    b = gamma * wildtype - W @ wildtype

    # ------------------------------------------------------------------
    # Gradient matching likelihood
    # For each timeseries, each timepoint:
    #   dx_i/dt ~ Normal(b_i + W[i,:]@x(t) - γ_i*x_i(t) + perturb_i(t), σ_gm_i)
    # ------------------------------------------------------------------
    all_x      = []   # stack all timepoints across all series
    all_dxdt   = []
    all_perturb = []

    for ts, dx_dt in zip(net.timeseries, dx_dt_list):
        x      = jnp.array(ts.expression,  dtype=jnp.float32)   # (T, N)
        dxdt   = jnp.array(dx_dt,          dtype=jnp.float32)   # (T, N)
        perturb = jnp.array(ts.perturbation, dtype=jnp.float32)  # (N,)
        T      = x.shape[0]

        # Perturbation applied for t < 500
        t_mask = jnp.array(ts.timepoints < 500.0, dtype=jnp.float32)  # (T,)
        perturb_t = t_mask[:, None] * perturb[None, :]                 # (T, N)

        all_x.append(x)
        all_dxdt.append(dxdt)
        all_perturb.append(perturb_t)

    X_all       = jnp.concatenate(all_x,       axis=0)   # (M, N)
    dXdt_all    = jnp.concatenate(all_dxdt,    axis=0)   # (M, N)
    perturb_all = jnp.concatenate(all_perturb, axis=0)   # (M, N)

    # Predicted derivative: (M, N)
    dx_pred = (b[None, :] + X_all @ W.T - X_all * gamma[None, :] + perturb_all)

    numpyro.sample(
        'obs_dx',
        dist.Normal(dx_pred, sigma_gm[None, :]),
        obs=dXdt_all,
    )

    # ------------------------------------------------------------------
    # Knockout steady-state likelihood (analytical, exact do-calculus)
    # ------------------------------------------------------------------
    if use_knockouts:
        sigma_ko = numpyro.sample('sigma_ko',
                                  dist.HalfNormal(sigma_obs_prior * jnp.ones(N)))
        for gene_idx, ko_expression in net.knockout_pairs:
            W_ko = W.at[:, gene_idx].set(0.0).at[gene_idx, :].set(0.0)
            b_ko = (gamma * wildtype - W_ko @ wildtype).at[gene_idx].set(0.0)
            A    = jnp.diag(gamma) - W_ko
            x_ss = jnp.linalg.solve(A, b_ko).at[gene_idx].set(0.0)
            numpyro.sample(
                f'obs_ko_{gene_idx}',
                dist.Normal(x_ss, sigma_ko),
                obs=jnp.array(ko_expression, dtype=jnp.float32),
            )


def run_phase3_gradient_matching(
    net: DREAM4Network,
    use_knockouts: bool = True,
    num_steps: int = 3000,
    lr: float = 0.01,
    tau0: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run gradient-matching MAP with regularised horseshoe prior.

    Returns
    -------
    edge_scores : (N, N)  |W_ij| — use for AUPR (diagonal = 0)
    W           : (N, N)  MAP edge weights
    gamma       : (N,)    MAP degradation rates
    losses      : SVI loss curve
    """
    dx_dt_list = _compute_gradients(net)

    def model():
        grn_model_gradient_matching(net, dx_dt_list,
                                    use_knockouts=use_knockouts, tau0=tau0)

    guide     = AutoDelta(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=1.0)
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

    if verbose:
        print(f'Phase 3 gradient-matching: {num_steps} steps, '
              f'lr={lr}, tau0={tau0}', flush=True)

    result = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=verbose)
    params = guide.median(result.params)

    N    = net.n_genes
    rows, cols = np.where(~np.eye(N, dtype=bool))
    lam  = np.array(params['lambda_ij'])
    c2   = float(params['c2'])
    tau  = float(params['tau'])
    lam_t = np.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
    w_od = np.array(params['w_raw']) * tau * lam_t
    W    = np.zeros((N, N))
    W[rows, cols] = w_od

    edge_scores = np.abs(W)
    np.fill_diagonal(edge_scores, 0.0)

    return {
        'edge_scores': edge_scores,
        'W':          W,
        'gamma':      np.array(params['gamma']),
        'losses':     np.array(result.losses),
    }


# ---------------------------------------------------------------------------
# Phase 3D: Knockout perturbation response (causal, no model required)
# ---------------------------------------------------------------------------

def run_phase3_knockout_response(
    net: DREAM4Network,
    deconvolve: bool = False,
    alpha: float = 0.9,
) -> dict:
    """Score edges using raw knockout perturbation responses.

    score[i, j] = |X_ko_i[j] - X_wt[j]|

    When gene i is knocked out, gene j's expression changes iff there is a
    direct or indirect regulatory path from i to j.  For sparse Size10
    networks this signal is highly informative (AUPR >> GENIE3).

    Parameters
    ----------
    deconvolve : bool
        If True, apply network deconvolution (Feizi et al. 2013) to remove
        indirect effects from the response matrix:
            S_direct ≈ S @ (I + S)^{-1}
        Scales S so the largest eigenvalue = alpha < 1 before inversion.
    alpha : float
        Spectral scaling factor for deconvolution (default 0.9).

    Returns
    -------
    edge_scores : (N, N)  score for edge i→j (diagonal = 0)
    """
    N  = net.n_genes
    wt = net.wildtype                      # (N,)
    scores = np.zeros((N, N), dtype=np.float64)

    for gene_idx, ko_expr in net.knockout_pairs:
        scores[gene_idx] = np.abs(ko_expr - wt)   # score for gene_idx → all others

    np.fill_diagonal(scores, 0.0)

    if deconvolve:
        # Network deconvolution: remove indirect paths from response matrix.
        # Scale so spectral radius < 1, then solve (I + S_scaled) X = S_scaled.
        ev_max = np.abs(np.linalg.eigvals(scores)).max()
        if ev_max > 0:
            S = alpha * scores / ev_max
        else:
            S = scores
        S_direct = np.linalg.solve(np.eye(N) + S, S)
        # Take absolute value (deconvolution can produce negatives)
        scores = np.abs(S_direct)
        np.fill_diagonal(scores, 0.0)

    return {'edge_scores': scores.astype(np.float32)}
