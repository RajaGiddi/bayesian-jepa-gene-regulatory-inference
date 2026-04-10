"""ODE dynamics model for the GRN World Model.

Linear SDE with basal transcription:
    dx_i/dt = b_i + sum_j G_ji * w_ji * x_j  -  gamma_i * x_i  +  perturbation_i(t)

where:
    G       : (N, N) binary adjacency matrix  (G_ji=1 means j regulates i)
    W       : (N, N) real-valued weight matrix (masked by G)
    gamma   : (N,)  degradation rates > 0
    b       : (N,)  basal transcription rates, derived from the wildtype constraint:
                    b_i = gamma_i * x_wt_i - (W ⊙ G @ x_wt)_i
                    This makes wildtype an exact steady state for any (W, G, gamma).
    perturbation : (N,) additional basal shift, nonzero during t in [0, t_perturb_end]

Intervention handling (exact do-calculus):
    Knockout gene k:  set x_k = 0 and zero out column k of G (no inputs to k)
                      also zero row k of G (k cannot regulate others when absent)
    Knockdown gene k: set basal_k to 0.5 * wildtype_k (halved transcription)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# DREAM4 perturbation schedule: applied at t=0, removed at t=500
T_PERTURB_END = 500.0
T_END = 1000.0


def compute_basal_rates(
    W: np.ndarray,
    G: np.ndarray,
    gamma: np.ndarray,
    wildtype: np.ndarray,
) -> np.ndarray:
    """Derive basal transcription rates from the wildtype steady-state constraint.

    At wildtype: dx/dt = 0, so:
        b_i = gamma_i * x_wt_i - (W ⊙ G @ x_wt)_i

    This guarantees wildtype is an exact steady state for any (W, G, gamma).

    Parameters
    ----------
    W        : (N, N) edge weight matrix
    G        : (N, N) binary adjacency mask
    gamma    : (N,) degradation rates
    wildtype : (N,) wildtype steady-state expression

    Returns
    -------
    b : (N,) basal transcription rates
    """
    return gamma * wildtype - (W * G) @ wildtype


def ode_rhs(
    t: float,
    x: np.ndarray,
    W_masked: np.ndarray,
    gamma: np.ndarray,
    b: np.ndarray,
    perturbation: np.ndarray,
    t_perturb_end: float = T_PERTURB_END,
) -> np.ndarray:
    """Right-hand side of the gene regulatory ODE.

    Parameters
    ----------
    t             : current time
    x             : (N,) current expression state
    W_masked      : (N, N) weight matrix already masked by G  (W * G)
    gamma         : (N,) degradation rates
    b             : (N,) basal transcription rates (from compute_basal_rates)
    perturbation  : (N,) additional basal shifts (nonzero during [0, t_perturb_end])
    t_perturb_end : time at which perturbation is removed (500 in DREAM4)

    Returns
    -------
    dxdt : (N,) derivative
    """
    shift = perturbation if t < t_perturb_end else np.zeros_like(perturbation)
    return b + W_masked @ x - gamma * x + shift


def ode_solve(
    W: np.ndarray,
    G: np.ndarray,
    gamma: np.ndarray,
    wildtype: np.ndarray,
    timepoints: np.ndarray,
    x0: np.ndarray,
    perturbation: np.ndarray,
    t_perturb_end: float = T_PERTURB_END,
) -> np.ndarray:
    """Solve the GRN ODE and return expression at observed timepoints.

    Parameters
    ----------
    W           : (N, N) edge weight matrix (sign = activation/repression)
    G           : (N, N) binary adjacency mask
    gamma       : (N,) degradation rates
    wildtype    : (N,) wildtype expression (used to derive basal rates)
    timepoints  : (T,) observation times
    x0          : (N,) initial expression state
    perturbation: (N,) additional basal shift applied during [0, t_perturb_end]

    Returns
    -------
    traj : (T, N) predicted expression at each timepoint
    """
    W_masked = W * G
    b = compute_basal_rates(W, G, gamma, wildtype)

    sol = solve_ivp(
        fun=lambda t, x: ode_rhs(t, x, W_masked, gamma, b, perturbation, t_perturb_end),
        t_span=(timepoints[0], timepoints[-1]),
        y0=x0,
        t_eval=timepoints,
        method="LSODA",   # handles both stiff and non-stiff automatically
        rtol=1e-4,
        atol=1e-6,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.y.T  # (T, N)


def apply_knockout(G: np.ndarray, gene_idx: int) -> np.ndarray:
    """Apply do(x_{gene_idx} = 0): remove all edges to/from knocked-out gene.

    Zeroes column gene_idx (no gene can be regulated by the KO'd gene)
    and row gene_idx (KO'd gene has no output / receives no inputs to integrate).

    Parameters
    ----------
    G        : (N, N) binary adjacency matrix
    gene_idx : index of gene to knock out

    Returns
    -------
    G_ko : (N, N) modified adjacency matrix (copy, G unchanged)
    """
    G_ko = G.copy()
    G_ko[:, gene_idx] = 0  # gene_idx cannot regulate others
    G_ko[gene_idx, :] = 0  # nothing regulates gene_idx (it's fixed at 0)
    return G_ko


def ode_steady_state(
    W: np.ndarray,
    G: np.ndarray,
    gamma: np.ndarray,
    wildtype: np.ndarray,
    x0: np.ndarray,
    knocked_out_gene: int | None = None,
    t_end: float = T_END,
) -> np.ndarray:
    """Run ODE to t_end and return final expression (approximate steady state).

    Used for knockout steady-state predictions: pass G_ko from apply_knockout
    and x0 = wildtype with the KO'd gene zeroed out.

    For correct knockout behaviour, pass knocked_out_gene so that gene's
    basal transcription rate is zeroed (otherwise b_k > 0 drives it back
    toward wildtype despite the knockout).
    """
    W_masked = W * G
    b = compute_basal_rates(W, G, gamma, wildtype)
    if knocked_out_gene is not None:
        b[knocked_out_gene] = 0.0  # no transcription of knocked-out gene

    perturbation = np.zeros(len(x0))

    sol = solve_ivp(
        fun=lambda t, x: ode_rhs(t, x, W_masked, gamma, b, perturbation, t_perturb_end=0.0),
        t_span=(0.0, t_end),
        y0=x0,
        t_eval=np.array([0.0, t_end]),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"ODE steady-state solver failed: {sol.message}")

    return sol.y.T[-1]  # (N,) final state


def validate_ode(
    net,
    W: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
) -> None:
    """Quick sanity check: run ODE on a DREAM4Network and print trajectory shape.

    If W and gamma are None, uses identity-scaled gold standard as W
    and ones as gamma — just to verify the solver runs without error.
    """
    N = net.n_genes

    if W is None:
        G = net.gold_standard if net.gold_standard is not None else np.eye(N)
        W = G * 0.1  # small weights
    if gamma is None:
        gamma = np.ones(N) * 0.5

    G = net.gold_standard if net.gold_standard is not None else np.eye(N)

    print(f"net{net.network_id} (Size{net.size}): {N} genes, {len(net.timeseries)} series")
    for i, ts in enumerate(net.timeseries):
        traj = ode_solve(W, G, gamma, net.wildtype, ts.timepoints, ts.expression[0], ts.perturbation)
        mse = np.mean((traj - ts.expression) ** 2)
        print(f"  Series {i+1}: traj shape={traj.shape}, MSE vs observed={mse:.4f}")

    print(f"  Knockouts: {len(net.knockout_pairs)} pairs")
    ko_gene, ko_ss = net.knockout_pairs[0]
    G_ko = apply_knockout(G, ko_gene)
    x0 = net.wildtype.copy()
    x0[ko_gene] = 0.0
    pred_ss = ode_steady_state(W, G_ko, gamma, net.wildtype, x0, knocked_out_gene=ko_gene)
    print(f"  KO gene G{ko_gene+1}: pred={pred_ss[:5].round(3)}, obs={ko_ss[:5].round(3)}")
