"""JAX/diffrax/numpyro implementation of the GRN World Model.

Replaces the PyMC + scipy ODE approach with:
  - diffrax  : JAX-native ODE solver (Tsit5), autodiff through solve
  - numpyro  : JIT-compiled NUTS, no sensitivity equations
  - jax.jit  : entire likelihood compiled to native code once

Speedup over PyMC + pm.ode.DifferentialEquation:
  ~30h (sensitivity equations, 260-dim augmented ODE)
  → ~30-45 min (autodiff through 10-dim ODE, JIT-compiled)

Knockout steady states are solved analytically (linear system) rather
than by long ODE integration — exact and fast.

Usage:
    import os; os.environ['JAX_PLATFORMS'] = 'cpu'
    from grn_world_model.jax_model import run_nuts

    trace = run_nuts(net.gold_standard, net, num_samples=500, num_warmup=500)
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value

from grn_world_model.data_loader import DREAM4Network
from grn_world_model.ode_model import apply_knockout

T_PERTURB_END = 500.0


# ---------------------------------------------------------------------------
# JAX ODE components
# ---------------------------------------------------------------------------

def _ode_rhs(t, y, args):
    """JAX-traceable ODE RHS for one timeseries.

    args = (W_masked, gamma, b, perturbation)
    dx/dt = b + W_masked @ x - gamma * x + shift(t)
    where shift(t) = perturbation if t < 500 else 0
    """
    W_masked, gamma, b, perturbation = args
    shift = jnp.where(t < T_PERTURB_END, perturbation, jnp.zeros_like(perturbation))
    return b + W_masked @ y - gamma * y + shift


def _solve_timeseries(
    W_masked: jnp.ndarray,
    gamma: jnp.ndarray,
    b: jnp.ndarray,
    x0: jnp.ndarray,
    t0: float,
    t1: float,
    save_ts: jnp.ndarray,
    perturbation: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ODE for one timeseries. Returns predicted expression (T-1, N).

    t0, t1 must be concrete Python floats (not JAX tracers) — diffrax
    requires these to be static at trace time.
    save_ts: JAX array of save times (timepoints[1:]).
    """
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_rhs),
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=1.0,
        y0=x0,
        args=(W_masked, gamma, b, perturbation),
        saveat=diffrax.SaveAt(ts=save_ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        max_steps=8192,  # fail fast in bad regions — 65536 was too slow to reject
        throw=False,     # return last valid state instead of crashing
    )
    return sol.ys


def _knockout_steady_state(
    W: jnp.ndarray,
    G_ko: jnp.ndarray,
    gamma: jnp.ndarray,
    wildtype: jnp.ndarray,
    gene_idx: int,
) -> jnp.ndarray:
    """Compute knockout steady state analytically (linear system solve).

    At steady state with no perturbation:
        (diag(gamma) - W_ko) @ x_ss = b_ko
    where b_ko = gamma * wt - W_ko @ wt  with b_ko[gene_idx] = 0.

    This is exact and O(N^3) — much faster than long ODE integration.
    Fully differentiable via jnp.linalg.solve.
    """
    W_ko = W * G_ko                                      # apply KO mask
    b = gamma * wildtype - W_ko @ wildtype
    b = b.at[gene_idx].set(0.0)                          # no transcription for KO gene
    A = jnp.diag(gamma) - W_ko                          # (N, N) system matrix
    # Add small ridge for numerical stability (gamma > 0 already makes A well-conditioned)
    x_ss = jnp.linalg.solve(A, b)
    x_ss = x_ss.at[gene_idx].set(0.0)                   # enforce KO gene = 0
    return x_ss


# ---------------------------------------------------------------------------
# Numpyro model
# ---------------------------------------------------------------------------

def grn_model(
    G_k: np.ndarray,
    net: DREAM4Network,
    use_knockouts: bool = True,
    use_timeseries: bool = False,
    w_prior_sigma: float = 0.3,
    gamma_prior_sigma: float = 1.0,
    sigma_obs_prior: float = 0.5,
    stability_coef: float = 5.0,
) -> None:
    """Numpyro model for GRN dynamics inference with fixed graph G_k.

    Priors:
        w_flat ~ Normal(0, w_prior_sigma)      per active edge
        gamma  ~ HalfNormal(gamma_prior_sigma) per gene (degradation rate)
        sigma  ~ HalfNormal(sigma_obs_prior)   per gene (obs noise)

    Likelihood:
        Knockouts (default ON):  analytical linear steady-state solve —
            smooth gradient, well-conditioned, NUTS-friendly.
        Timeseries (default OFF): ODE integration via diffrax — concentrates
            the posterior heavily but makes NUTS unmixable due to exponential
            sensitivity of ODE solutions to parameters. Use MAP instead.
    """
    N = net.n_genes
    active_edges = np.argwhere(G_k)
    n_active = len(active_edges)

    G_k_jnp  = jnp.array(G_k, dtype=jnp.float32)
    wildtype = jnp.array(net.wildtype, dtype=jnp.float32)

    if use_knockouts:
        G_ko_list = [
            jnp.array(apply_knockout(G_k, gene_idx), dtype=jnp.float32)
            for gene_idx, _ in net.knockout_pairs
        ]

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------
    w_flat = numpyro.sample(
        'w_flat',
        dist.Normal(jnp.zeros(n_active), w_prior_sigma * jnp.ones(n_active)),
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
    # Build W matrix from active edges
    # ------------------------------------------------------------------
    W = jnp.zeros((N, N))
    if n_active > 0:
        W = W.at[active_edges[:, 0], active_edges[:, 1]].set(w_flat)
    W_masked = W * G_k_jnp

    # Gershgorin stability: penalise regions where ODE can blow up.
    # Applies during both MAP (prevents Adam divergence) and NUTS.
    if stability_coef > 0:
        row_excess = jnp.sum(jnp.abs(W_masked), axis=1) - gamma
        numpyro.factor('stability',
                       -stability_coef * jnp.sum(jnp.maximum(row_excess, 0.0)))

    # Basal rates: b_i = gamma_i * wt_i - (W_masked @ wt)_i
    b = gamma * wildtype - W_masked @ wildtype

    # ------------------------------------------------------------------
    # Timeseries likelihood (ODE-based — off by default for NUTS)
    # ------------------------------------------------------------------
    if use_timeseries:
        for i, ts in enumerate(net.timeseries):
            t0_f        = float(ts.timepoints[0])
            t1_f        = float(ts.timepoints[-1])
            save_ts     = jnp.array(ts.timepoints[1:], dtype=jnp.float32)
            x0          = jnp.array(ts.expression[0], dtype=jnp.float32)
            perturbation = jnp.array(ts.perturbation, dtype=jnp.float32)
            observed    = jnp.array(ts.expression[1:], dtype=jnp.float32)
            x_pred = _solve_timeseries(W_masked, gamma, b, x0, t0_f, t1_f,
                                       save_ts, perturbation)
            numpyro.sample(
                f'obs_ts_{i}',
                dist.Normal(x_pred, sigma_obs[None, :]),
                obs=observed,
            )

    # ------------------------------------------------------------------
    # Knockout steady-state likelihood (analytical linear solve)
    # ------------------------------------------------------------------
    if use_knockouts:
        for (gene_idx, ko_expression), G_ko in zip(net.knockout_pairs, G_ko_list):
            x_ss = _knockout_steady_state(W, G_ko, gamma, wildtype, gene_idx)
            numpyro.sample(
                f'obs_ko_{gene_idx}',
                dist.Normal(x_ss, sigma_obs),
                obs=jnp.array(ko_expression, dtype=jnp.float32),
            )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def run_nuts(
    G_k: np.ndarray,
    net: DREAM4Network,
    num_samples: int = 500,
    num_warmup: int = 500,
    num_chains: int = 2,
    use_knockouts: bool = True,
    use_timeseries: bool = False,
    seed: int = 42,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    **model_kwargs,
) -> dict:
    """Run NUTS on the fixed-graph model. Returns numpyro MCMC object.

    Default: knockout-only (use_timeseries=False). The analytical linear
    steady-state likelihood is smooth and well-conditioned — NUTS mixes well.
    The ODE timeseries likelihood concentrates the posterior so tightly
    (~1000 observations) that NUTS step sizes collapse; use MAP for that.

    Parameters
    ----------
    G_k               : (N, N) fixed binary adjacency matrix
    net               : DREAM4Network
    num_samples       : posterior draws per chain
    num_warmup        : warmup (tuning) steps per chain
    num_chains        : number of chains (run sequentially on CPU)
    use_knockouts     : include knockout likelihood (analytical, default True)
    use_timeseries    : include ODE timeseries likelihood (default False)
    target_accept_prob: dual-averaging target
    max_tree_depth    : max NUTS tree depth (default 10)

    Returns
    -------
    mcmc : numpyro MCMC object (call .print_summary() or extract samples)
    """
    def model():
        grn_model(G_k, net, use_knockouts=use_knockouts,
                  use_timeseries=use_timeseries, **model_kwargs)

    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_median(num_samples=20),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    numpyro.set_host_device_count(num_chains)  # suppress sequential-chain warning
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def jax_map(
    G_k: np.ndarray,
    net: DREAM4Network,
    use_knockouts: bool = True,
    use_timeseries: bool = True,
    num_steps: int = 2000,
    lr: float = 0.01,
    seed: int = 0,
    **model_kwargs,
) -> dict:
    """Gradient-based MAP via numpyro SVI + Adam (AutoDelta guide).

    Unlike pymc Powell, this uses autodiff through the full likelihood and
    never gets stuck on flat plateaus. Returns a dict of parameter arrays
    in constrained space (same format as pymc map_estimate).

    use_timeseries=True uses the diffrax ODE; use_timeseries=False is
    knockout-only (analytical, always stable).
    """
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta

    def model():
        grn_model(G_k, net, use_knockouts=use_knockouts,
                  use_timeseries=use_timeseries, **model_kwargs)

    guide     = AutoDelta(model)
    optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=1.0)
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())
    result    = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=False)

    # Extract MAP point in constrained space
    params = guide.median(result.params)
    return {k: np.array(v) for k, v in params.items()}


def mcmc_to_arviz(mcmc: MCMC) -> object:
    """Convert numpyro MCMC samples to ArviZ InferenceData for diagnostics."""
    import arviz as az
    return az.from_numpyro(mcmc)
