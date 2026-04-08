"""Ablation 5: Inverted JEPA masking.

Standard B-JEPA: TF expression (context) → predict all gene representations.
Inverted masking: non-TF gene expression (context) → predict TF representations.

Intuition for GRN inference
---------------------------
If TF_t regulates gene_g, then gene_g's expression is a readout of TF_t's
activity.  Regressing TF expression on non-TF gene expression captures this
indirect signal in the OPPOSITE direction:

    score(TF_t → gene_g) ≈ coeff of gene_g when predicting TF_t

This is complementary to forward regression (which regresses gene_g on TFs).
The two directions provide independent evidence that a TF–gene pair is
regulatory, so ensembling should improve precision.

Analytical implementation (GCV ridge)
--------------------------------------
Forward  : X_tf  (n, D)     → regress Y_genes (n, G)   → W (D, G)
Inverted : X_gene (n, G−D)  → regress Y_tf   (n, D)    → W_inv (G−D, D)

Edge scores from inverted: W_inv.T → (D, G−D) = (n_tfs, n_nontf_genes)

Two variants:
    Global GCV  — single α* minimising total RSS across all D TF targets
    Per-TF GCV  — individual α_t* per TF target (mirrors Ablation 4)

TF→TF edges are not covered by the inverted direction (TF genes are removed
from the context set).  They are filled with 0 so that ensembling with the
forward GCV scores naturally compensates.

Usage:
    python experiments/ablations/inverted_masking.py            # all networks
    python experiments/ablations/inverted_masking.py --network 2
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.eval.metrics import evaluate_predictions

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "ablations" / "inverted_masking"
GCV_DIR     = ROOT / "results" / "analytical_hs" / "ols"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["score"].rank(method="average") / len(df)
    return df


def ensemble(a: pd.DataFrame, b: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """alpha · a_score + (1−alpha) · b_score.  Both must be rank-normalised."""
    m = a.merge(b, on=["tf", "target"], suffixes=("_a", "_b"))
    m["score"] = alpha * m["score_a"] + (1 - alpha) * m["score_b"]
    return m[["tf", "target", "score"]]


# ---------------------------------------------------------------------------
# Core: inverted GCV ridge
# ---------------------------------------------------------------------------

def inverted_gcv_scores(
    network,
    n_alpha: int = 80,
    per_tf: bool = True,
    horseshoe: bool = True,
    p0: float = 10.0,
) -> pd.DataFrame:
    """Inverted masking: regress TF expression on non-TF gene expression.

    Parameters
    ----------
    network  : DREAM5Network
    n_alpha  : number of ridge α values on log-grid
    per_tf   : if True, use per-TF GCV (α_t* per TF); else global α*
    horseshoe: apply horseshoe κ shrinkage on top of ridge
    p0       : expected nonzero non-TF genes per TF (for horseshoe τ₀)

    Returns
    -------
    DataFrame [tf, target, score] with TF→non-TF edges only.
    TF→TF edges are absent (caller may pad or rely on ensemble with forward scores).
    """
    expr = network.expression.values.astype(np.float64)   # (n, G)
    n, G = expr.shape

    gene_mean = expr.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr.std(axis=0, keepdims=True), 1e-8)
    expr_std  = (expr - gene_mean) / gene_std              # (n, G)

    gene_ids   = network.gene_ids
    tf_set     = set(network.tf_ids)
    tf_ids     = network.tf_ids
    tf_indices = [i for i, g in enumerate(gene_ids) if g in tf_set]
    ng_indices = [i for i, g in enumerate(gene_ids) if g not in tf_set]

    # Context (predictor): non-TF genes
    X = expr_std[:, ng_indices]   # (n, G−D)
    # Targets: TF genes
    Y = expr_std[:, tf_indices]   # (n, D)
    M, D = X.shape[1], Y.shape[1]  # M = n_nontf, D = n_tfs

    # ----------------------------------------------------------------
    # SVD of X  — O(n · M²)
    # ----------------------------------------------------------------
    U, d, Vt = np.linalg.svd(X, full_matrices=False)  # d: (min(n,M),)
    d2  = d ** 2                                        # (K,)
    UtY = U.T @ Y                                       # (K, D)

    # ----------------------------------------------------------------
    # α grid — log-spaced from 1e-4 to 1e4·n
    # ----------------------------------------------------------------
    alpha_grid = np.logspace(-4, math.log10(1e4 * n), n_alpha)

    # ----------------------------------------------------------------
    # RSS and denominator grids
    # rss_grid  : (n_alpha, D)
    # denom_grid: (n_alpha,)  same for all TF targets
    # ----------------------------------------------------------------
    rss_grid   = np.empty((n_alpha, D), dtype=np.float64)
    denom_grid = np.empty(n_alpha,      dtype=np.float64)

    for i, alpha in enumerate(alpha_grid):
        shrink    = d2 / (d2 + alpha)                     # (K,)
        trace_H   = shrink.sum()
        Y_hat     = U @ (shrink[:, None] * UtY)           # (n, D)
        residuals = Y - Y_hat
        rss_grid[i]   = (residuals ** 2).sum(axis=0)      # (D,)
        denom_grid[i] = n * (1.0 - trace_H / n) ** 2

    # ----------------------------------------------------------------
    # Choose α (global or per-TF)
    # ----------------------------------------------------------------
    gcv_grid = rss_grid / denom_grid[:, None]             # (n_alpha, D)

    if per_tf:
        # Per-TF: argmin over α for each TF independently
        best_idx    = gcv_grid.argmin(axis=0)             # (D,)
        alpha_star  = alpha_grid[best_idx]                # (D,)

        # Compute W_inv column by column (grouped by best_idx for efficiency)
        K = d.shape[0]
        W_inv = np.empty((M, D), dtype=np.float64)        # (G−D, D)
        for alpha_idx in np.unique(best_idx):
            alpha  = alpha_grid[alpha_idx]
            shrink = d / (d2 + alpha)                     # (K,)
            tmask  = (best_idx == alpha_idx)
            W_inv[:, tmask] = Vt.T @ (shrink[:, None] * UtY[:, tmask])
    else:
        # Global: single α* minimising total GCV across all TF targets
        total_gcv   = gcv_grid.sum(axis=1)               # (n_alpha,)
        best_i      = int(total_gcv.argmin())
        alpha_star  = alpha_grid[best_i] * np.ones(D)
        alpha       = alpha_grid[best_i]
        shrink      = d / (d2 + alpha)                   # (K,)
        W_inv       = Vt.T @ (shrink[:, None] * UtY)    # (M, D)

    # ----------------------------------------------------------------
    # Optional horseshoe shrinkage (κ filter)
    # ----------------------------------------------------------------
    if horseshoe:
        # Per-TF noise variance from residuals at chosen α
        trace_H_t  = np.array([(d2 / (d2 + alpha_star[t])).sum() for t in range(D)])
        df_eff_t   = np.maximum(n - trace_H_t, 1.0)
        rss_best_t = rss_grid[gcv_grid.argmin(axis=0), np.arange(D)]  # (D,)
        sigma2_t   = rss_best_t / df_eff_t
        sigma_t    = np.sqrt(sigma2_t)

        # tau0 per TF: p0 / (M − p0) * sigma_t / sqrt(n)
        tau0_t    = (p0 / max(M - p0, 1.0)) * sigma_t / math.sqrt(n)  # (D,)
        beta2     = W_inv ** 2                                          # (M, D)
        prior_var = sigma2_t[None, :] * tau0_t[None, :] ** 2           # (1, D)
        kappa     = 1.0 / (1.0 + n * beta2 / np.maximum(prior_var, 1e-20))
        abs_score = np.abs((1.0 - kappa) * W_inv)                      # (M, D)
    else:
        abs_score = np.abs(W_inv)                                       # (M, D)

    # ----------------------------------------------------------------
    # Edge scores: W_inv.T[t, g] = score for edge TF_t → gene_g (g ∈ non-TF)
    # ----------------------------------------------------------------
    # abs_score.T : (D, M) = (n_tfs, n_nontf)
    scores_T = abs_score.T

    tf_arr  = np.array(tf_ids)
    ng_arr  = np.array([gene_ids[i] for i in ng_indices])

    tf_names   = np.repeat(tf_arr[:, None], M, axis=1)   # (D, M)
    gene_names = np.repeat(ng_arr[None, :], D, axis=0)   # (D, M)
    not_self   = tf_names != gene_names                   # (D, M) always True here

    df = pd.DataFrame({
        "tf":     tf_names[not_self],
        "target": gene_names[not_self],
        "score":  scores_T[not_self],
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(nid: int) -> dict:
    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    print(f"\n{'='*60}")
    print(f"Ablation 5 — Inverted masking  |  net{nid} ({network.name})")
    print(f"  n={network.n_samples}  D={network.n_tfs}  G={network.n_genes}"
          f"  n_nontf={network.n_genes - network.n_tfs}"
          f"  n/D={network.n_samples/network.n_tfs:.2f}")

    # Forward GCV baseline
    gcv_path = GCV_DIR / f"net{nid}" / "edge_scores.csv"
    gcv_df   = pd.read_csv(gcv_path)
    gcv_df["score"] = gcv_df["score"].abs()
    gcv_m  = evaluate_predictions(gcv_df, network.gold_standard)
    print(f"\n  Forward GCV-Ridge AUPR = {gcv_m['aupr']:.4f}  (baseline)")

    # ----------------------------------------------------------------
    # Inverted masking: global α and per-TF α, with/without horseshoe
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    inv_global = inverted_gcv_scores(network, per_tf=False, horseshoe=False)
    inv_global_hs = inverted_gcv_scores(network, per_tf=False, horseshoe=True)
    inv_pertf  = inverted_gcv_scores(network, per_tf=True,  horseshoe=False)
    inv_pertf_hs = inverted_gcv_scores(network, per_tf=True, horseshoe=True)
    elapsed = time.perf_counter() - t0

    m_glob   = evaluate_predictions(inv_global,    network.gold_standard)
    m_glob_hs = evaluate_predictions(inv_global_hs, network.gold_standard)
    m_ptf    = evaluate_predictions(inv_pertf,     network.gold_standard)
    m_ptf_hs = evaluate_predictions(inv_pertf_hs,  network.gold_standard)

    print(f"\n  Inverted (global α, no HS):  AUPR={m_glob['aupr']:.4f}")
    print(f"  Inverted (global α + HS):    AUPR={m_glob_hs['aupr']:.4f}")
    print(f"  Inverted (per-TF α, no HS):  AUPR={m_ptf['aupr']:.4f}")
    print(f"  Inverted (per-TF α + HS):    AUPR={m_ptf_hs['aupr']:.4f}  ({elapsed:.1f}s)")

    # ----------------------------------------------------------------
    # Ensemble: forward GCV + best inverted variant
    # Choose the best inverted variant (per-TF + HS expected to be best)
    # ----------------------------------------------------------------
    # Build a complete edge DataFrame for the inverted scores.
    # TF→TF edges are absent from inverted; fill them with 0 so that
    # rank-normalisation treats them as lowest scores in the inverted set.
    all_edges = gcv_df[["tf", "target"]].drop_duplicates().copy()
    inv_best  = inv_pertf_hs.copy()  # best variant

    # Left-join all edges with inverted scores; fill missing with 0
    inv_full = all_edges.merge(
        inv_best.rename(columns={"score": "score_inv"}),
        on=["tf", "target"], how="left"
    ).fillna({"score_inv": 0.0})
    inv_full = inv_full.rename(columns={"score_inv": "score"})

    gcv_rn  = rank_normalise(gcv_df)
    inv_rn  = rank_normalise(inv_full)

    best_ens = {"aupr": -1.0, "alpha": -1.0}
    for alpha in np.linspace(0.0, 1.0, 21):
        m = evaluate_predictions(ensemble(gcv_rn, inv_rn, alpha),
                                 network.gold_standard)
        if m["aupr"] > best_ens["aupr"]:
            best_ens = {"aupr": m["aupr"], "alpha": round(float(alpha), 2)}

    print(f"\n  Ensemble GCV + Inverted(best): AUPR={best_ens['aupr']:.4f}"
          f"  (α_fwd={best_ens['alpha']})")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n  {'Method':<35} {'AUPR':>8} {'Δ vs GCV':>10}")
    print(f"  {'-'*55}")
    for label, aupr in [
        ("Forward GCV-Ridge",           gcv_m["aupr"]),
        ("Inverted (global α)",          m_glob["aupr"]),
        ("Inverted (global α + HS)",     m_glob_hs["aupr"]),
        ("Inverted (per-TF α)",          m_ptf["aupr"]),
        ("Inverted (per-TF α + HS)",     m_ptf_hs["aupr"]),
        ("Ensemble GCV + Inverted",       best_ens["aupr"]),
    ]:
        delta = aupr - gcv_m["aupr"]
        print(f"  {label:<35} {aupr:>8.4f} {delta:>+10.4f}")

    # Save
    out_dir = RESULTS_DIR / f"net{nid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    inv_pertf_hs.to_csv(out_dir / "inverted_scores.csv", index=False)
    print(f"\n  Saved to {out_dir}/")

    return {
        "network":           f"net{nid} ({network.name})",
        "gcv_aupr":          gcv_m["aupr"],
        "inv_global_aupr":   m_glob["aupr"],
        "inv_global_hs":     m_glob_hs["aupr"],
        "inv_pertf_aupr":    m_ptf["aupr"],
        "inv_pertf_hs":      m_ptf_hs["aupr"],
        "ensemble_aupr":     best_ens["aupr"],
        "best_alpha_fwd":    best_ens["alpha"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]
    results = []
    for nid in nids:
        r = run(nid)
        if r:
            results.append(r)

    if len(results) > 1:
        print("\n\nSummary across networks:")
        df = pd.DataFrame(results).set_index("network")
        print(df.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
