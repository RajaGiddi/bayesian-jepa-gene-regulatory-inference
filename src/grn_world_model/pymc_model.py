"""Per-particle PyMC model for the GRN World Model.

Given a fixed graph G_k (binary adjacency matrix), fit the dynamics
parameters (W, gamma, sigma_obs) using NUTS or ADVI.

The likelihood has two components:
  1. Timeseries: Gaussian likelihood over ODE-predicted trajectories
  2. Knockouts:  Gaussian likelihood over knockout steady-state predictions

Note on gradients: the scipy ODE solver is not differentiable through
pytensor. We use blackbox Ops (finite-difference gradients disabled) and
rely on pm.find_MAP() or ADVI for fast fitting, and NUTS with
nuts_sampler="numpyro" for full posteriors on Size10 networks.
For gradient-based NUTS, replace with pm.ode.DifferentialEquation.
"""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

from grn_world_model.data_loader import DREAM4Network
from grn_world_model.ode_model import apply_knockout, ode_solve, ode_steady_state


# ---------------------------------------------------------------------------
# Helpers: build W matrix from flat active-edge weights
# ---------------------------------------------------------------------------

def _build_W(w_flat: np.ndarray, active_edges: np.ndarray, N: int) -> np.ndarray:
    """Reconstruct (N, N) weight matrix from flat active-edge weights."""
    W = np.zeros((N, N), dtype=np.float64)
    if len(active_edges):
        W[active_edges[:, 0], active_edges[:, 1]] = w_flat
    return W


# ---------------------------------------------------------------------------
# Blackbox Ops (no gradient — MAP / ADVI only)
# ---------------------------------------------------------------------------

class _TimeseriesOp(Op):
    """Blackbox Op: (w_flat, gamma) → predicted trajectory (T, N)."""

    __props__ = ()

    def __init__(self, G_k, active_edges, N, wildtype, ts):
        self.G_k = G_k
        self.active_edges = active_edges
        self.N = N
        self.wildtype = wildtype
        self.ts = ts

    def make_node(self, w_flat, gamma):
        w_flat = pt.as_tensor_variable(w_flat)
        gamma  = pt.as_tensor_variable(gamma)
        out    = pt.dmatrix()
        return Apply(self, [w_flat, gamma], [out])

    def perform(self, node, inputs, outputs):
        w_flat, gamma = inputs
        W = _build_W(w_flat, self.active_edges, self.N)
        try:
            traj = ode_solve(
                W, self.G_k, gamma, self.wildtype,
                self.ts.timepoints, self.ts.expression[0], self.ts.perturbation,
            )
        except (RuntimeError, ValueError):
            # ODE blew up — return observed values so log-likelihood ≈ 0
            # (neutral, not -inf) and the optimizer backs away
            traj = self.ts.expression.copy()
        outputs[0][0] = traj.astype(np.float64)


class _KnockoutOp(Op):
    """Blackbox Op: (w_flat, gamma) → knockout steady-state (N,)."""

    __props__ = ()

    def __init__(self, G_k, active_edges, N, wildtype, gene_idx):
        self.G_k = G_k
        self.active_edges = active_edges
        self.N = N
        self.wildtype = wildtype
        self.gene_idx = gene_idx
        self.G_ko = apply_knockout(G_k, gene_idx)

    def make_node(self, w_flat, gamma):
        w_flat = pt.as_tensor_variable(w_flat)
        gamma  = pt.as_tensor_variable(gamma)
        out    = pt.dvector()
        return Apply(self, [w_flat, gamma], [out])

    def perform(self, node, inputs, outputs):
        w_flat, gamma = inputs
        W = _build_W(w_flat, self.active_edges, self.N)
        x0 = self.wildtype.copy()
        x0[self.gene_idx] = 0.0
        try:
            ss = ode_steady_state(
                W, self.G_ko, gamma, self.wildtype, x0,
                knocked_out_gene=self.gene_idx,
            )
        except (RuntimeError, ValueError):
            ss = self.wildtype.copy()
            ss[self.gene_idx] = 0.0
        outputs[0][0] = ss.astype(np.float64)


# ---------------------------------------------------------------------------
# PyMC model builder
# ---------------------------------------------------------------------------

def build_per_particle_model(
    G_k: np.ndarray,
    net: DREAM4Network,
    use_knockouts: bool = True,
    sigma_obs_prior: float = 0.5,
    w_prior_sigma: float = 0.3,
    gamma_prior_sigma: float = 1.0,
) -> pm.Model:
    """Build PyMC model for dynamics inference given fixed graph G_k.

    Parameters
    ----------
    G_k              : (N, N) binary adjacency matrix
    net              : DREAM4Network with wildtype, timeseries, knockouts
    use_knockouts    : include knockout steady-state likelihood terms
    sigma_obs_prior  : HalfNormal scale for observation noise
    w_prior_sigma    : Normal scale for edge weights
    gamma_prior_sigma: HalfNormal scale for degradation rates

    Returns
    -------
    pm.Model ready for pm.find_MAP(), pm.fit(), or pm.sample()
    """
    N = net.n_genes
    active_edges = np.argwhere(G_k)   # (E, 2)
    n_active = len(active_edges)
    wildtype = net.wildtype            # (N,)

    with pm.Model() as model:

        # ------------------------------------------------------------------
        # Priors
        # ------------------------------------------------------------------
        w_flat = pm.Normal(
            "w_flat", mu=0.0, sigma=w_prior_sigma,
            shape=(n_active,) if n_active > 0 else (1,),
        )
        gamma = pm.HalfNormal("gamma", sigma=gamma_prior_sigma, shape=N)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma_obs_prior, shape=N)

        # ------------------------------------------------------------------
        # Timeseries likelihood
        # ------------------------------------------------------------------
        for i, ts in enumerate(net.timeseries):
            op = _TimeseriesOp(G_k, active_edges, N, wildtype, ts)
            x_pred = pm.Deterministic(f"x_pred_{i}", op(w_flat, gamma))
            # x_pred: (T, N), ts.expression: (T, N)
            pm.Normal(
                f"obs_ts_{i}",
                mu=x_pred,
                sigma=sigma_obs[None, :],
                observed=ts.expression.astype(np.float64),
            )

        # ------------------------------------------------------------------
        # Knockout likelihood (exact do(x_k = 0) interventions)
        # ------------------------------------------------------------------
        if use_knockouts:
            for gene_idx, ko_expression in net.knockout_pairs:
                op = _KnockoutOp(G_k, active_edges, N, wildtype, gene_idx)
                x_ko_pred = pm.Deterministic(f"x_ko_{gene_idx}", op(w_flat, gamma))
                pm.Normal(
                    f"obs_ko_{gene_idx}",
                    mu=x_ko_pred,
                    sigma=sigma_obs,
                    observed=ko_expression.astype(np.float64),
                )

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def map_estimate(model: pm.Model, maxeval: int = 3000) -> dict:
    """Find MAP estimate via Powell (gradient-free). Capped at maxeval evals."""
    with model:
        return pm.find_MAP(progressbar=True, maxeval=maxeval)


def sample_advi(
    model: pm.Model,
    n_iter: int = 30_000,
    n_samples: int = 2_000,
) -> tuple:
    """Run ADVI (fast, no ODE gradients needed).

    Suitable for Size100 where NUTS is intractable.
    Returns (approx, trace).
    """
    with model:
        approx = pm.fit(n=n_iter, method="advi", progressbar=True)
        trace = approx.sample(n_samples)
    return approx, trace


def sample_nuts(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 4,
    target_accept: float = 0.9,
    nuts_sampler: str = "numpyro",
) -> object:
    """Run NUTS. Use for Size10 (10 genes) only — slow for larger networks.

    Note: blackbox Ops don't support NUTS gradients. Use this only after
    replacing Ops with pm.ode.DifferentialEquation, or use ADVI instead.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            return_inferencedata=True,
        )
    return trace


# ---------------------------------------------------------------------------
# Phase 2b: NUTS model using pm.ode.DifferentialEquation
# Provides adjoint-method gradients through the ODE — required for NUTS.
# ---------------------------------------------------------------------------

def _make_ts_ode_func(active_edges, N, G_k_arr, wildtype_np, perturbation_np):
    """Return a pytensor-traceable ODE function for one timeseries.

    The function signature func(y, t, p) is traced once by DifferentialEquation.
    t is a pytensor scalar — use pt.switch for piecewise perturbation.
    p = [w_flat (n_active), gamma (N)]
    """
    n_active = len(active_edges)
    G_const = G_k_arr.astype(np.float64)
    wt = wildtype_np.astype(np.float64)
    perturb = perturbation_np.astype(np.float64)

    def ode_func(y, t, p):
        w_f = p[:n_active]
        gam = p[n_active:n_active + N]

        # Build masked weight matrix
        W = pt.zeros((N, N))
        if n_active > 0:
            W = pt.set_subtensor(
                W[active_edges[:, 0], active_edges[:, 1]], w_f
            )
        W_masked = W * G_const                         # (N, N), G is numpy const

        # Basal rates: b_i = gamma_i * wt_i - (W*G @ wt)_i
        b = gam * wt - pt.dot(W_masked, wt)

        # Piecewise perturbation: active for t < 500
        shift = pt.switch(
            pt.lt(t, 500.0),
            pt.as_tensor_variable(perturb),
            pt.zeros(N),
        )

        return b + pt.dot(W_masked, y) - gam * y + shift

    return ode_func


def _make_ko_ode_func(active_edges, N, G_ko_arr, wildtype_np, gene_idx):
    """Return ODE function for knockout steady-state integration.

    b[gene_idx] is zeroed (no basal transcription for KO'd gene).
    No perturbation (integrating to steady state).
    """
    n_active = len(active_edges)
    G_const = G_ko_arr.astype(np.float64)
    wt = wildtype_np.astype(np.float64)

    def ko_ode_func(y, t, p):
        w_f = p[:n_active]
        gam = p[n_active:n_active + N]

        W = pt.zeros((N, N))
        if n_active > 0:
            W = pt.set_subtensor(
                W[active_edges[:, 0], active_edges[:, 1]], w_f
            )
        W_masked = W * G_const

        b = gam * wt - pt.dot(W_masked, wt)
        b = pt.set_subtensor(b[gene_idx], 0.0)  # no transcription for KO'd gene

        return b + pt.dot(W_masked, y) - gam * y

    return ko_ode_func


def build_nuts_model(
    G_k: np.ndarray,
    net: DREAM4Network,
    use_knockouts: bool = True,
    sigma_obs_prior: float = 0.5,
    w_prior_sigma: float = 0.3,
    gamma_prior_sigma: float = 1.0,
    ko_t_end: float = 2000.0,
) -> pm.Model:
    """Build PyMC model using pm.ode.DifferentialEquation for NUTS compatibility.

    Identical likelihood to build_per_particle_model but uses the adjoint
    method for gradients, enabling gradient-based samplers (NUTS, ADVI).

    Parameters
    ----------
    G_k           : (N, N) fixed binary adjacency matrix
    net           : DREAM4Network
    use_knockouts : include N knockout steady-state likelihood terms
    ko_t_end      : integration time for knockout steady state (default 2000)

    Returns
    -------
    pm.Model ready for pm.sample() with NUTS.
    """
    N = net.n_genes
    active_edges = np.argwhere(G_k)
    n_active = len(active_edges)
    n_theta = n_active + N
    wildtype = net.wildtype

    # Build one DifferentialEquation per timeseries (fixed func per series)
    ts_odes = []
    for ts in net.timeseries:
        func = _make_ts_ode_func(active_edges, N, G_k, wildtype, ts.perturbation)
        ode = pm.ode.DifferentialEquation(
            func=func,
            times=ts.timepoints[1:].tolist(),   # exclude t=0 (initial condition)
            n_states=N,
            n_theta=n_theta,
        )
        ts_odes.append((ode, ts))

    # Build one DifferentialEquation per knockout
    ko_odes = []
    if use_knockouts:
        for gene_idx, ko_expression in net.knockout_pairs:
            G_ko = apply_knockout(G_k, gene_idx)
            func = _make_ko_ode_func(active_edges, N, G_ko, wildtype, gene_idx)
            ode = pm.ode.DifferentialEquation(
                func=func,
                times=[ko_t_end],
                n_states=N,
                n_theta=n_theta,
            )
            x0_ko = wildtype.copy()
            x0_ko[gene_idx] = 0.0
            ko_odes.append((ode, gene_idx, ko_expression, x0_ko))

    with pm.Model() as model:

        # Priors
        w_flat = pm.Normal(
            "w_flat", mu=0.0, sigma=w_prior_sigma,
            shape=(n_active,) if n_active > 0 else (1,),
        )
        gamma = pm.HalfNormal("gamma", sigma=gamma_prior_sigma, shape=N)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma_obs_prior, shape=N)

        theta = pm.Deterministic("theta", pt.concatenate([w_flat, gamma]))

        # Timeseries likelihood
        for i, (ode, ts) in enumerate(ts_odes):
            x_pred = pm.Deterministic(
                f"x_pred_{i}",
                ode(y0=ts.expression[0].tolist(), theta=theta),
            )
            pm.Normal(
                f"obs_ts_{i}",
                mu=x_pred,
                sigma=sigma_obs[None, :],
                observed=ts.expression[1:].astype(np.float64),
            )

        # Knockout steady-state likelihood
        for ode, gene_idx, ko_exp, x0_ko in ko_odes:
            x_ko_pred = pm.Deterministic(
                f"x_ko_{gene_idx}",
                ode(y0=x0_ko.tolist(), theta=theta)[0],  # [0]: single timepoint → (N,)
            )
            pm.Normal(
                f"obs_ko_{gene_idx}",
                mu=x_ko_pred,
                sigma=sigma_obs,
                observed=ko_exp.astype(np.float64),
            )

    return model


def sample_nuts_ode(
    G_k: np.ndarray,
    net: DREAM4Network,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    target_accept: float = 0.9,
    **model_kwargs,
) -> tuple:
    """Build NUTS model and sample. Returns (model, trace).

    Use for Size10 networks only (10 genes, ~15 active edges).
    Runtime: ~5–30 min depending on n_active and chain length.
    """
    model = build_nuts_model(G_k, net, **model_kwargs)
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True,
        )
    return model, trace
