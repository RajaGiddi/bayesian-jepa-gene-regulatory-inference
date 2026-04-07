"""Analytical horseshoe GRN inference — DREAM5 benchmark runner.

Runs two variants per network:
  - OLS mode:  pure OLS + horseshoe shrinkage (no JEPA warm start)
  - MAP mode:  MAP estimate with W_init prior mean (JEPA warm start)

No gradient descent required. Both variants run in seconds.

Usage (from repo root):

    python experiments/exp3_analytical_hs/run.py              # all networks
    python experiments/exp3_analytical_hs/run.py --network 1  # single network
    python experiments/exp3_analytical_hs/run.py --alpha 500  # override MAP alpha
    python experiments/exp3_analytical_hs/run.py --p0 20      # override p0

Results written to results/analytical_hs/{ols,map}/.
Prints full comparison table: GENIE3 / OLS-HS / MAP-HS / B-JEPA-VI.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.models.analytical_horseshoe import analytical_horseshoe_scores
from bjepa.eval.metrics import evaluate_predictions

DATA_DIR         = ROOT / "data"
RESULTS_DIR      = ROOT / "results" / "analytical_hs"
BJEPA_RESULTS    = ROOT / "results" / "bjepa"
GENIE3_SUMMARY   = ROOT / "results" / "genie3" / "summary.json"

DEFAULT_P0 = {1: 10, 2: 10, 3: 10, 4: 10}

# Default MAP alpha per network (n = number of samples).
# Expressed as a multiplier of n: alpha = alpha_scale * n.
# Larger → solution pulled harder toward W_init.
DEFAULT_ALPHA_SCALE = {1: 1.0, 2: 10.0, 3: 1.0, 4: 1.0}

# Ridge factor for pure OLS mode.
# For well-determined networks (n >> p) use 1e-6 (numerical stability only).
# For underdetermined/ill-conditioned networks (net2: n/p=1.6) use larger value
# so the OLS estimate is stabilized — the VI horseshoe's KL does this implicitly.
# Expressed as a fraction of n: delta = ridge_factor * n.
DEFAULT_RIDGE = {1: 1e-6, 2: 0.05, 3: 1e-6, 4: 0.05}


def load_W_init(nid: int) -> np.ndarray | None:
    """Load W_init from the saved Stage 1 checkpoint, or None if missing."""
    ckpt_path = BJEPA_RESULTS / f"net{nid}" / "stage1_final.pt"
    if not ckpt_path.exists():
        return None
    sd = torch.load(ckpt_path, map_location="cpu")
    if "W_init" not in sd:
        return None
    return sd["W_init"].numpy().astype(np.float64)  # (n_tfs, n_genes)


def _report_shrinkage_stats(scores_df: pd.DataFrame, label: str) -> None:
    scores = scores_df["score"].values
    p99 = np.percentile(scores, 99)
    p50 = np.percentile(scores, 50)
    frac_nonzero = (scores > 1e-8).mean()
    print(f"    [{label}] p99={p99:.4f}  p50={p50:.4f}  frac_nonzero={frac_nonzero:.3f}")


def run_network(
    nid: int,
    p0: float | None = None,
    alpha_scale: float | None = None,
    ridge_factor: float | None = None,
) -> dict:
    net = load_network(DATA_DIR, nid)
    if net.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    _p0          = p0 if p0 is not None else DEFAULT_P0[nid]
    _alpha_scale = alpha_scale if alpha_scale is not None else DEFAULT_ALPHA_SCALE[nid]
    _ridge       = ridge_factor if ridge_factor is not None else DEFAULT_RIDGE[nid]
    alpha        = _alpha_scale * net.n_samples

    print(f"\n{'='*60}")
    print(f"net{nid} ({net.name})  "
          f"n={net.n_samples} p={net.n_tfs} genes={net.n_genes}  "
          f"p0={_p0}  ridge={_ridge}  alpha={alpha:.0f} ({_alpha_scale}·n)")
    print(f"{'='*60}")

    out_ols = RESULTS_DIR / "ols" / f"net{nid}"
    out_map = RESULTS_DIR / "map" / f"net{nid}"
    out_ols.mkdir(parents=True, exist_ok=True)
    out_map.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Ridge-only ablation (no horseshoe shrinkage)
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    scores_ridge = analytical_horseshoe_scores(net, p0=_p0, ridge_factor=_ridge, horseshoe=False)
    elapsed_ridge = time.perf_counter() - t0
    m_ridge = evaluate_predictions(scores_ridge, net.gold_standard)
    print(f"  Ridge:AUROC={m_ridge['auroc']:.4f}  AUPR={m_ridge['aupr']:.4f}  ({elapsed_ridge:.1f}s)")

    # ----------------------------------------------------------------
    # OLS-HS mode (ridge + horseshoe shrinkage)
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    scores_ols = analytical_horseshoe_scores(net, p0=_p0, ridge_factor=_ridge)
    elapsed_ols = time.perf_counter() - t0
    m_ols = evaluate_predictions(scores_ols, net.gold_standard)
    print(f"  OLS-HS:AUROC={m_ols['auroc']:.4f}  AUPR={m_ols['aupr']:.4f}  ({elapsed_ols:.1f}s)")
    _report_shrinkage_stats(scores_ols, "OLS-HS")
    scores_ols.to_csv(out_ols / "edge_scores.csv", index=False)

    # ----------------------------------------------------------------
    # MAP mode (W_init warm start from Stage 1)
    # ----------------------------------------------------------------
    W_init = load_W_init(nid)
    m_map  = {"auroc": float("nan"), "aupr": float("nan")}
    elapsed_map = 0.0

    if W_init is not None:
        t0 = time.perf_counter()
        scores_map = analytical_horseshoe_scores(
            net, p0=_p0, W_prior_mean=W_init, alpha=alpha
        )
        elapsed_map = time.perf_counter() - t0
        m_map = evaluate_predictions(scores_map, net.gold_standard)
        print(f"  MAP:  AUROC={m_map['auroc']:.4f}  AUPR={m_map['aupr']:.4f}  ({elapsed_map:.1f}s)")
        _report_shrinkage_stats(scores_map, "MAP")
        scores_map.to_csv(out_map / "edge_scores.csv", index=False)
    else:
        print(f"  MAP:  stage1_final.pt not found for net{nid}, skipping MAP mode.")

    result = {
        "network":          f"net{nid} ({net.name})",
        "ridge_auroc":      m_ridge["auroc"],
        "ridge_aupr":       m_ridge["aupr"],
        "ols_auroc":        m_ols["auroc"],
        "ols_aupr":         m_ols["aupr"],
        "map_auroc":        m_map["auroc"],
        "map_aupr":         m_map["aupr"],
        "alpha":            alpha,
        "p0":               _p0,
        "n_samples":        net.n_samples,
        "n_tfs":            net.n_tfs,
        "n_genes":          net.n_genes,
        "n_positive_edges": net.n_positive_edges,
    }
    with open(out_ols / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def print_comparison(results: list[dict]) -> None:
    if not results:
        return

    res_df = pd.DataFrame(results).set_index("network")

    # Load GENIE3 results
    g3 = None
    if GENIE3_SUMMARY.exists():
        with open(GENIE3_SUMMARY) as f:
            g3 = pd.DataFrame(json.load(f)).set_index("network")

    # Load B-JEPA VI results from per-network result.json (not stale summary.json)
    bj_rows = []
    for nid in [1, 2, 3, 4]:
        rj = BJEPA_RESULTS / f"net{nid}" / "result.json"
        if rj.exists():
            with open(rj) as f:
                bj_rows.append(json.load(f))
    bj = pd.DataFrame(bj_rows).set_index("network") if bj_rows else None

    rows = {}
    for net in res_df.index:
        rows[net] = {}
        if g3 is not None and net in g3.index:
            rows[net]["GENIE3 AUROC"] = g3.loc[net, "auroc"]
            rows[net]["GENIE3 AUPR"]  = g3.loc[net, "aupr"]
        rows[net]["Ridge AUROC"]   = res_df.loc[net, "ridge_auroc"]
        rows[net]["Ridge AUPR"]    = res_df.loc[net, "ridge_aupr"]
        rows[net]["OLS-HS AUROC"]  = res_df.loc[net, "ols_auroc"]
        rows[net]["OLS-HS AUPR"]   = res_df.loc[net, "ols_aupr"]
        rows[net]["MAP-HS AUROC"]  = res_df.loc[net, "map_auroc"]
        rows[net]["MAP-HS AUPR"]   = res_df.loc[net, "map_aupr"]
        if bj is not None and net in bj.index:
            rows[net]["VI+W_init AUROC"] = bj.loc[net, "auroc"]
            rows[net]["VI+W_init AUPR"]  = bj.loc[net, "aupr"]

    cmp = pd.DataFrame(rows).T
    print("\n\nFull comparison table:")
    print(cmp.to_string(float_format="{:.4f}".format))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",     type=int,   choices=[1, 2, 3, 4])
    parser.add_argument("--p0",          type=float, default=None)
    parser.add_argument("--alpha_scale", type=float, default=None,
                        help="MAP prior precision as multiple of n. Default: per-network.")
    parser.add_argument("--ridge_factor", type=float, default=None,
                        help="OLS ridge as fraction of n (e.g. 0.05 stabilises net2). Default: per-network.")
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]

    all_results = []
    for nid in nids:
        result = run_network(
            nid, p0=args.p0,
            alpha_scale=args.alpha_scale,
            ridge_factor=args.ridge_factor,
        )
        if result:
            all_results.append(result)

    if all_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).set_index("network").to_csv(RESULTS_DIR / "summary.csv")
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print_comparison(all_results)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
