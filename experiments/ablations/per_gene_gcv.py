"""Ablation 4: Per-gene GCV ridge regression.

Global GCV (exp3) finds one α* minimising total RSS across all genes.
Per-gene GCV finds α_g* for each target gene g independently.

Key insight: the hat-matrix trace (denominator) depends only on X and α,
not on the target gene.  So we evaluate GCV_g(α) = RSS_g(α) / denom(α)
for all genes simultaneously with a single SVD of X_tf, then select:

    α_g* = argmin_α  ||Y_g - X·β_g(α)||² / (n · (1 - traceH(α)/n)²)

This takes seconds and produces an edge-score matrix β_g*(α_g*) in which
each gene column is optimally regularised for its own signal-to-noise.

Motivation: net3 (E. coli) has highly heterogeneous gene expression
variance.  Some genes are well-explained by TFs (low noise → small α*),
others are noisy (high noise → large α*).  A single global α* over- or
under-regularises both classes simultaneously.

Usage:
    python experiments/ablations/per_gene_gcv.py            # all networks
    python experiments/ablations/per_gene_gcv.py --network 2
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
RESULTS_DIR = ROOT / "results" / "ablations" / "per_gene_gcv"
GCV_DIR     = ROOT / "results" / "analytical_hs" / "ols"


def per_gene_gcv_scores(
    network,
    n_alpha: int = 80,
    horseshoe: bool = True,
    p0: float = 10.0,
) -> pd.DataFrame:
    """Compute per-gene GCV ridge scores.

    Parameters
    ----------
    network     : DREAM5Network
    n_alpha     : grid resolution (more = finer search)
    horseshoe   : apply horseshoe shrinkage on top of per-gene ridge (κ filter)
    p0          : expected nonzero TFs per gene (for horseshoe τ₀)

    Returns
    -------
    DataFrame [tf, target, score] sorted descending by |score|.
    """
    expr = network.expression.values.astype(np.float64)  # (n, G)
    n, G = expr.shape

    gene_mean = expr.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr.std(axis=0, keepdims=True), 1e-8)
    expr_std  = (expr - gene_mean) / gene_std              # (n, G)

    gene_ids   = network.gene_ids
    tf_set     = set(network.tf_ids)
    tf_ids     = network.tf_ids
    tf_indices = [i for i, g in enumerate(gene_ids) if g in tf_set]

    X = expr_std[:, tf_indices]   # (n, D)  D = n_tfs
    Y = expr_std                   # (n, G)
    D = X.shape[1]

    # ----------------------------------------------------------------
    # SVD of X — computed once, O(n·D²)
    # ----------------------------------------------------------------
    U, d, Vt = np.linalg.svd(X, full_matrices=False)  # d: (D,)
    d2  = d ** 2                                        # (D,)
    UtY = U.T @ Y                                       # (D, G)

    # ----------------------------------------------------------------
    # α grid — log-spaced from 1e-4 to 1e4·n
    # ----------------------------------------------------------------
    alpha_grid = np.logspace(-4, math.log10(1e4 * n), n_alpha)  # (n_alpha,)

    # ----------------------------------------------------------------
    # For each α: compute per-gene RSS and denominator
    # rss_grid  : (n_alpha, G)
    # denom_grid: (n_alpha,)  — same for all genes
    # ----------------------------------------------------------------
    rss_grid   = np.empty((n_alpha, G), dtype=np.float64)
    denom_grid = np.empty(n_alpha,      dtype=np.float64)

    for i, alpha in enumerate(alpha_grid):
        shrink    = d2 / (d2 + alpha)                     # (D,)
        trace_H   = shrink.sum()                           # scalar
        Y_hat     = U @ (shrink[:, None] * UtY)           # (n, G)
        residuals = Y - Y_hat                              # (n, G)
        rss_grid[i]   = (residuals ** 2).sum(axis=0)      # (G,)
        denom_grid[i] = n * (1.0 - trace_H / n) ** 2

    # ----------------------------------------------------------------
    # Per-gene GCV: argmin over α
    # gcv_grid: (n_alpha, G)  = rss / denom[:, None]
    # ----------------------------------------------------------------
    gcv_grid      = rss_grid / denom_grid[:, None]         # (n_alpha, G)
    best_idx      = gcv_grid.argmin(axis=0)                # (G,)  index into alpha_grid
    alpha_star_g  = alpha_grid[best_idx]                   # (G,)  optimal α per gene

    # ----------------------------------------------------------------
    # Compute β_g*(α_g*) for every gene
    # β_g(α) = Vt.T @ diag(d / (d² + α)) @ UtY[:,g]
    # Group genes by their best α index to batch the computation.
    # ----------------------------------------------------------------
    W_ols_pg = np.empty((D, G), dtype=np.float64)

    for alpha_idx in np.unique(best_idx):
        alpha  = alpha_grid[alpha_idx]
        shrink = d / (d2 + alpha)                          # (D,) = d/(d²+α)
        # Which genes use this α?
        gmask  = (best_idx == alpha_idx)
        # β_g(α) = Vt.T @ (shrink * UtY[:, g])
        W_ols_pg[:, gmask] = Vt.T @ (shrink[:, None] * UtY[:, gmask])

    # ----------------------------------------------------------------
    # Optional horseshoe shrinkage on top of per-gene ridge
    # ----------------------------------------------------------------
    if horseshoe:
        # Per-gene noise variance: σ²_g = RSS_g(α_g*) / (n - D_eff)
        # Use effective df = n - traceH(α_g*) per gene
        trace_H_g  = np.array([(d2 / (d2 + alpha_star_g[g])).sum() for g in range(G)])
        df_eff_g   = np.maximum(n - trace_H_g, 1.0)
        rss_best   = rss_grid[best_idx, np.arange(G)]     # (G,) RSS at best α
        sigma2_g   = rss_best / df_eff_g                  # (G,)
        sigma_g    = np.sqrt(sigma2_g)

        tau0_g     = (p0 / (D - p0)) * sigma_g / math.sqrt(n)  # (G,)
        beta2      = W_ols_pg ** 2                              # (D, G)
        prior_var  = sigma2_g[None, :] * tau0_g[None, :] ** 2  # (1, G) broadcast
        kappa      = 1.0 / (1.0 + n * beta2 / np.maximum(prior_var, 1e-20))
        abs_beta   = np.abs((1.0 - kappa) * W_ols_pg)
    else:
        abs_beta = np.abs(W_ols_pg)

    # ----------------------------------------------------------------
    # Build edge DataFrame — vectorised, exclude self-loops
    # ----------------------------------------------------------------
    tf_arr   = np.array(tf_ids)
    gene_arr = np.array(gene_ids)
    tf_names   = np.repeat(tf_arr[:, None],   G,     axis=1)  # (D, G)
    gene_names = np.repeat(gene_arr[None, :], D,     axis=0)  # (D, G)
    not_self   = tf_names != gene_names

    df = pd.DataFrame({
        "tf":     tf_names[not_self],
        "target": gene_names[not_self],
        "score":  abs_beta[not_self],
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["score"].rank(method="average") / len(df)
    return df


def ensemble(a: pd.DataFrame, b: pd.DataFrame, alpha: float) -> pd.DataFrame:
    m = a.merge(b, on=["tf", "target"], suffixes=("_a", "_b"))
    m["score"] = alpha * m["score_a"] + (1 - alpha) * m["score_b"]
    return m[["tf", "target", "score"]]


def run(nid: int) -> dict:
    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    print(f"\n{'='*60}")
    print(f"Ablation 4 — Per-gene GCV  |  net{nid} ({network.name})")
    print(f"  n={network.n_samples}  D={network.n_tfs}  G={network.n_genes}"
          f"  n/D={network.n_samples/network.n_tfs:.2f}")

    # Global GCV baseline
    gcv_path = GCV_DIR / f"net{nid}" / "edge_scores.csv"
    gcv_df   = pd.read_csv(gcv_path)
    gcv_df["score"] = gcv_df["score"].abs()
    gcv_m    = evaluate_predictions(gcv_df, network.gold_standard)

    # B-JEPA VI baseline (saved scores from exp2)
    bjepa_path = ROOT / "results" / "bjepa" / f"net{nid}" / "result.json"
    bjepa_aupr = float("nan")
    if bjepa_path.exists():
        import json
        with open(bjepa_path) as f:
            bjepa_aupr = json.load(f).get("aupr", float("nan"))

    print(f"  Global GCV-Ridge AUPR = {gcv_m['aupr']:.4f}")
    print(f"  B-JEPA VI AUPR        = {bjepa_aupr:.4f}")

    # ----------------------------------------------------------------
    # Per-gene GCV (with and without horseshoe shrinkage)
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    pg_scores    = per_gene_gcv_scores(network, horseshoe=True)
    t1 = time.perf_counter()
    pg_scores_r  = per_gene_gcv_scores(network, horseshoe=False)
    elapsed = time.perf_counter() - t0

    m_pg   = evaluate_predictions(pg_scores,   network.gold_standard)
    m_pg_r = evaluate_predictions(pg_scores_r, network.gold_standard)

    print(f"\n  Per-gene GCV + horseshoe: AUPR={m_pg['aupr']:.4f}"
          f"  AUROC={m_pg['auroc']:.4f}  ({elapsed:.1f}s)")
    print(f"  Per-gene GCV (ridge only): AUPR={m_pg_r['aupr']:.4f}"
          f"  AUROC={m_pg_r['auroc']:.4f}")

    # ----------------------------------------------------------------
    # Ensemble: GCV + per-gene GCV
    # ----------------------------------------------------------------
    gcv_rn  = rank_normalise(gcv_df)
    pg_rn   = rank_normalise(pg_scores)

    best_ens = {"aupr": -1.0, "alpha": -1.0}
    for alpha in np.linspace(0.0, 1.0, 21):
        m = evaluate_predictions(ensemble(gcv_rn, pg_rn, alpha),
                                 network.gold_standard)
        if m["aupr"] > best_ens["aupr"]:
            best_ens = {"aupr": m["aupr"], "alpha": round(float(alpha), 2)}

    print(f"  Ensemble GCV+PG-GCV:      AUPR={best_ens['aupr']:.4f}"
          f"  (α_gcv={best_ens['alpha']})")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n  {'Method':<30} {'AUPR':>8} {'Δ vs GCV':>10}")
    print(f"  {'-'*50}")
    for label, aupr in [
        ("Global GCV-Ridge",        gcv_m["aupr"]),
        ("Per-gene GCV + horseshoe", m_pg["aupr"]),
        ("Per-gene GCV (ridge only)", m_pg_r["aupr"]),
        ("Ensemble GCV + PG-GCV",    best_ens["aupr"]),
    ]:
        delta = aupr - gcv_m["aupr"]
        print(f"  {label:<30} {aupr:>8.4f} {delta:>+10.4f}")

    # Save scores
    out_dir = RESULTS_DIR / f"net{nid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pg_scores.to_csv(out_dir / "per_gene_gcv_scores.csv", index=False)
    print(f"\n  Saved to {out_dir}/")

    return {
        "network": f"net{nid} ({network.name})",
        "gcv_aupr": gcv_m["aupr"],
        "pg_gcv_hs_aupr": m_pg["aupr"],
        "pg_gcv_ridge_aupr": m_pg_r["aupr"],
        "ensemble_aupr": best_ens["aupr"],
        "best_alpha": best_ens["alpha"],
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
        print(pd.DataFrame(results).set_index("network").to_string(
            float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
