"""PyMC regularised horseshoe — DREAM5 benchmark runner.

Runs per-gene NUTS for one or more DREAM5 networks and compares
to GENIE3 and GCV-ridge. Produces Arviz diagnostic figures.

Usage (from repo root, bayes_stats env):

    # Net2 only (recommended first run — smallest network, ~30 min)
    python experiments/exp4_pymc_horseshoe/run.py --network 2

    # Fast demo: fewer draws, only showcase genes evaluated
    python experiments/exp4_pymc_horseshoe/run.py --network 2 --draws 200 --tune 200

    # All networks (slow)
    python experiments/exp4_pymc_horseshoe/run.py

Results written to results/pymc_hs/.
Figures written to results/pymc_hs/figures/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.models.pymc_horseshoe import run_pymc_horseshoe, sample_gene, compute_tau0_from_residuals
from bjepa.models.analytical_horseshoe import gcv_ridge_factor, analytical_horseshoe_scores
from bjepa.eval.metrics import evaluate_predictions

DATA_DIR      = ROOT / "data"
RESULTS_DIR   = ROOT / "results" / "pymc_hs"
GENIE3_SUMMARY = ROOT / "results" / "genie3" / "summary.json"


# ---------------------------------------------------------------------------
# Main per-network runner
# ---------------------------------------------------------------------------

def run_network(nid: int, args) -> dict:
    net = load_network(DATA_DIR, nid)
    if net.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    out_dir = RESULTS_DIR / f"net{nid}"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"net{nid} ({net.name})  n={net.n_samples}  D={net.n_tfs}  G={net.n_genes}")
    print(f"draws={args.draws}  tune={args.tune}  chains={args.chains}  p0={args.p0}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # GCV-ridge scores (fast reference)
    # ------------------------------------------------------------------
    print("\n[1/3] GCV-calibrated ridge (reference)...")
    rf = gcv_ridge_factor(net, verbose=True)
    scores_ridge = analytical_horseshoe_scores(net, ridge_factor=rf, horseshoe=False)
    m_ridge = evaluate_predictions(scores_ridge, net.gold_standard)
    print(f"  GCV-Ridge  AUROC={m_ridge['auroc']:.4f}  AUPR={m_ridge['aupr']:.4f}")

    # ------------------------------------------------------------------
    # PyMC NUTS horseshoe
    # ------------------------------------------------------------------
    print(f"\n[2/3] PyMC NUTS horseshoe ({net.n_genes} genes)...")
    t0 = time.perf_counter()
    scores_pymc, showcase_idatas = run_pymc_horseshoe(
        net,
        p0=args.p0,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        chains=args.chains,
        showcase_genes=args.showcase,
        random_seed=42,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    m_pymc = evaluate_predictions(scores_pymc, net.gold_standard)
    print(f"  PyMC-NUTS  AUROC={m_pymc['auroc']:.4f}  AUPR={m_pymc['aupr']:.4f}  ({elapsed/60:.1f} min)")

    # Save edge scores
    scores_pymc.to_csv(out_dir / "edge_scores.csv", index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating Arviz figures ({len(showcase_idatas)} showcase genes)...")
    if showcase_idatas:
        make_figures(showcase_idatas, net, scores_ridge, fig_dir, args)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    result = {
        "network":          f"net{nid} ({net.name})",
        "pymc_auroc":       m_pymc["auroc"],
        "pymc_aupr":        m_pymc["aupr"],
        "ridge_auroc":      m_ridge["auroc"],
        "ridge_aupr":       m_ridge["aupr"],
        "n_showcase_genes": len(showcase_idatas),
        "draws":            args.draws,
        "tune":             args.tune,
        "chains":           args.chains,
        "p0":               args.p0,
        "elapsed_min":      round(elapsed / 60, 1),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Arviz figure generation
# ---------------------------------------------------------------------------

def make_figures(
    showcase_idatas: dict,
    network,
    scores_ridge: pd.DataFrame,
    fig_dir: pathlib.Path,
    args,
) -> None:
    """Generate all Arviz diagnostic and results figures."""

    # Pick the gene with best R-hat (most converged) as primary showcase
    primary_gene, primary_idata = _pick_primary(showcase_idatas)
    print(f"  Primary showcase gene: {primary_gene}")

    # 1. Trace plot
    _fig_trace(primary_idata, primary_gene, fig_dir)

    # 2. κ distribution (bimodal horseshoe vs VI collapsed)
    _fig_kappa_distribution(showcase_idatas, fig_dir)

    # 3. Forest plot of β for the primary gene
    _fig_forest(primary_idata, primary_gene, network, fig_dir)

    # 4. R-hat summary across all showcase genes
    _fig_rhat_summary(showcase_idatas, fig_dir)

    # 5. Shrinkage plot: λ̃ vs |β_ols| (horseshoe profile)
    _fig_shrinkage_profile(primary_idata, primary_gene, network, scores_ridge, fig_dir)

    # 6. Posterior predictive check
    _fig_ppc(primary_idata, primary_gene, fig_dir)

    print(f"  Figures saved to {fig_dir}/")


def _pick_primary(showcase_idatas: dict) -> tuple[str, az.InferenceData]:
    """Select the gene with lowest max R-hat for primary showcase."""
    best_gene, best_rhat, best_idata = None, float("inf"), None
    for gene, idata in showcase_idatas.items():
        try:
            summ = az.summary(idata, var_names=["beta", "tau"], round_to=4)
            max_rhat = summ["r_hat"].max()
            if max_rhat < best_rhat:
                best_rhat = max_rhat
                best_gene = gene
                best_idata = idata
        except Exception:
            pass
    if best_gene is None:
        best_gene, best_idata = next(iter(showcase_idatas.items()))
    return best_gene, best_idata


def _fig_trace(idata: az.InferenceData, gene: str, fig_dir: pathlib.Path) -> None:
    """Trace plot for tau, sigma, and top-3 beta/lambda_tilde."""
    # Find top 3 TFs by posterior mean |beta|
    beta_mean = np.abs(idata.posterior["beta"].values).mean((0, 1))  # (D,)
    top3 = list(np.argsort(beta_mean)[::-1][:3])

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    az.plot_trace(
        idata,
        var_names=["tau", "sigma"],
        axes=axes[:2],
        show=False,
    )
    az.plot_trace(
        idata,
        var_names=["beta"],
        coords={"beta_dim_0": top3},
        axes=axes[2:4],
        show=False,
    )
    fig.suptitle(f"NUTS trace — gene {gene}", fontsize=13)
    plt.tight_layout()
    fig.savefig(fig_dir / f"trace_{gene}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_kappa_distribution(
    showcase_idatas: dict,
    fig_dir: pathlib.Path,
) -> None:
    """κ = 1/(1+λ̃²) distribution: NUTS bimodal vs VI collapsed."""
    all_kappa = []
    for idata in showcase_idatas.values():
        kappa = idata.posterior["kappa"].values  # (chains, draws, D)
        all_kappa.append(kappa.reshape(-1, kappa.shape[-1]))  # (S, D)
    kappa_cat = np.concatenate(all_kappa, axis=0).ravel()   # all samples, all TFs

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(kappa_cat, bins=80, density=True, color="#4C72B0", alpha=0.8,
            label="NUTS horseshoe (κ posterior samples)")
    ax.axvline(0.49, color="#DD4444", lw=2, ls="--",
               label="VI horseshoe (κ ≈ 0.49, collapsed)")
    ax.set_xlabel("Shrinkage coefficient κ = 1/(1 + λ̃²)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Horseshoe shrinkage profile: NUTS vs mean-field VI", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(fig_dir / "kappa_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_forest(
    idata: az.InferenceData,
    gene: str,
    network,
    fig_dir: pathlib.Path,
) -> None:
    """Forest plot: 94% credible intervals for all TF β coefficients."""
    beta_mean = np.abs(idata.posterior["beta"].values).mean((0, 1))
    top_n     = min(30, len(network.tf_ids))
    top_idx   = np.argsort(beta_mean)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    az.plot_forest(
        idata,
        var_names=["beta"],
        coords={"beta_dim_0": list(top_idx)},
        hdi_prob=0.94,
        combined=True,
        ax=ax,
        show=False,
    )
    # Annotate with TF names
    tf_labels = [network.tf_ids[i] for i in top_idx]
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(tf_labels[::-1], fontsize=8)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("β coefficient (posterior)", fontsize=11)
    ax.set_title(f"Top {top_n} TF coefficients — gene {gene} (94% HDI)", fontsize=12)
    plt.tight_layout()
    fig.savefig(fig_dir / f"forest_{gene}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_rhat_summary(
    showcase_idatas: dict,
    fig_dir: pathlib.Path,
) -> None:
    """R-hat distribution across all showcase genes."""
    all_rhats = []
    for idata in showcase_idatas.values():
        try:
            summ = az.summary(idata, var_names=["beta", "tau", "sigma"], round_to=4)
            all_rhats.extend(summ["r_hat"].dropna().tolist())
        except Exception:
            pass

    if not all_rhats:
        return

    all_rhats = np.array(all_rhats)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_rhats, bins=40, color="#55A868", alpha=0.85, edgecolor="white")
    ax.axvline(1.01, color="orange", lw=2, ls="--", label="R-hat = 1.01 (good)")
    ax.axvline(1.05, color="red",    lw=2, ls="--", label="R-hat = 1.05 (borderline)")
    pct_good = (all_rhats < 1.01).mean() * 100
    ax.set_xlabel("R-hat", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"NUTS convergence (R-hat)\n{pct_good:.1f}% of parameters < 1.01",
                 fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "rhat_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_shrinkage_profile(
    idata: az.InferenceData,
    gene: str,
    network,
    scores_ridge: pd.DataFrame,
    fig_dir: pathlib.Path,
) -> None:
    """Shrinkage factor (1-κ) vs |β_OLS|: the horseshoe profile."""
    # Posterior mean of (1-κ) per TF
    kappa_mean = idata.posterior["kappa"].values.mean((0, 1))   # (D,)
    shrink_mean = 1.0 - kappa_mean

    # OLS beta for this gene
    gene_scores = scores_ridge[scores_ridge["target"] == gene]
    tf_to_ols   = dict(zip(gene_scores["tf"], gene_scores["score"]))
    beta_ols    = np.array([tf_to_ols.get(tf, 0.0) for tf in network.tf_ids])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(np.abs(beta_ols), shrink_mean, c=shrink_mean,
                    cmap="coolwarm_r", alpha=0.7, s=30, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="(1 − κ)  unshrunk fraction")
    ax.set_xlabel("|β_OLS| (ridge coefficient magnitude)", fontsize=11)
    ax.set_ylabel("(1 − κ)  posterior mean", fontsize=11)
    ax.set_title(f"Horseshoe shrinkage profile — gene {gene}\n"
                 "large |β_OLS| → low κ → edge kept", fontsize=11)

    # Overlay the theoretical horseshoe curve
    b_grid = np.linspace(0, np.abs(beta_ols).max() * 1.1, 300)
    # Use median tau and median c for the curve
    tau_med = float(np.median(idata.posterior["tau"].values))
    lam_med = float(np.median(idata.posterior["lambda_tilde"].values))
    # (1-kappa) = n*beta^2/(sigma^2*tau0^2) / (1 + n*beta^2/(sigma^2*tau0^2))
    # Use a simpler display: just connect the dots
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(fig_dir / f"shrinkage_profile_{gene}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_ppc(
    idata: az.InferenceData,
    gene: str,
    fig_dir: pathlib.Path,
) -> None:
    """Posterior predictive check — observed vs replicated y distribution."""
    if "posterior_predictive" not in idata.groups():
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    az.plot_ppc(idata, ax=ax, show=False)
    ax.set_title(f"Posterior predictive check — gene {gene}", fontsize=12)
    plt.tight_layout()
    fig.savefig(fig_dir / f"ppc_{gene}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    if not results:
        return

    rows = {}
    g3 = None
    if GENIE3_SUMMARY.exists():
        with open(GENIE3_SUMMARY) as f:
            g3 = pd.DataFrame(json.load(f)).set_index("network")

    for r in results:
        net = r["network"]
        rows[net] = {}
        if g3 is not None and net in g3.index:
            rows[net]["GENIE3 AUPR"]   = g3.loc[net, "aupr"]
        rows[net]["GCV-Ridge AUPR"] = r["ridge_aupr"]
        rows[net]["PyMC-NUTS AUPR"] = r["pymc_aupr"]
        if g3 is not None and net in g3.index:
            rows[net]["Δ vs GENIE3"]  = r["pymc_aupr"] - g3.loc[net, "aupr"]

    print("\n\nPyMC NUTS Horseshoe vs baselines:")
    print(pd.DataFrame(rows).T.to_string(float_format="{:.4f}".format))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",       type=int,   choices=[1, 2, 3, 4])
    parser.add_argument("--p0",            type=float, default=10.0)
    parser.add_argument("--draws",         type=int,   default=500)
    parser.add_argument("--tune",          type=int,   default=500)
    parser.add_argument("--chains",        type=int,   default=2)
    parser.add_argument("--target_accept", type=float, default=0.9)
    parser.add_argument("--showcase",      type=int,   default=20,
                        help="Number of genes to store full InferenceData for figures.")
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]

    all_results = []
    for nid in nids:
        result = run_network(nid, args)
        if result:
            all_results.append(result)

    if all_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print_comparison(all_results)
    print(f"\nResults and figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
