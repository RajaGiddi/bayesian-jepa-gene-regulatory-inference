"""Generate all paper figures from saved results.

Figures that require only saved results (no retraining needed):
  Fig 1 — AUPR vs n/TF ratio: GCV-ridge crossover over GENIE3
  Fig 2 — Ablation bar chart: all methods on net2 and net3
  Fig 3 — GCV calibration curves: GCV score vs alpha for each network
  Fig 4 — VI horseshoe lambda collapse: kappa histogram from saved VI checkpoint
  Fig 5 — Method comparison table heatmap

Figures that require PyMC NUTS results (run exp4 first):
  Fig 6 — kappa distribution: NUTS bimodal vs VI collapsed  (from results/pymc_hs/)
  Fig 7 — Trace plot          (from results/pymc_hs/)
  Fig 8 — Forest plot         (from results/pymc_hs/)
  Fig 9 — R-hat summary       (from results/pymc_hs/)

Usage (from repo root):

    python experiments/figures/make_paper_figures.py            # all available figs
    python experiments/figures/make_paper_figures.py --no-pymc  # skip PyMC figures
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT      = pathlib.Path(__file__).resolve().parents[2]
FIG_DIR   = ROOT / "results" / "figures"
import sys; sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.models.analytical_horseshoe import gcv_ridge_factor

DATA_DIR        = ROOT / "data"
GENIE3_SUMMARY  = ROOT / "results" / "genie3" / "summary.json"
BJEPA_RESULTS   = ROOT / "results" / "bjepa"
ANA_HS_RESULTS  = ROOT / "results" / "analytical_hs"
PYMC_RESULTS    = ROOT / "results" / "pymc_hs"

# Consistent colours across all figures
COLOURS = {
    "GENIE3":      "#E07B39",
    "GCV-Ridge":   "#4C72B0",
    "VI+KL":       "#55A868",
    "MAP+W_init":  "#C44E52",
    "PyMC-NUTS":   "#8172B2",
}

NETWORK_LABELS = {
    "net1 (in_silico)":       "net1\n(in silico)",
    "net2 (s_aureus)":        "net2\n(S. aureus)",
    "net3 (e_coli)":          "net3\n(E. coli)",
    "net4 (s_cerevisiae)":    "net4\n(S. cerevisiae)",
}

# Ground-truth summary numbers (from individual result.json files)
GENIE3_AUPR  = {1: 0.2585, 2: 0.0112, 3: 0.0993, 4: 0.0221}
GENIE3_AUROC = {1: 0.8300, 2: 0.6778, 3: 0.6942, 4: 0.5412}
N_OVER_TF    = {1: 4.13,   2: 1.62,   3: 2.41,   4: 1.61}
GCV_AUPR     = {1: 0.2371, 2: 0.0175, 3: 0.1423, 4: 0.0241}
GCV_AUROC    = {1: 0.7505, 2: 0.6504, 3: 0.6573, 4: 0.5414}
VI_AUPR      = {1: 0.2035, 2: 0.0164, 3: 0.0799, 4: None}
VI_AUROC     = {1: 0.7439, 2: 0.5994, 3: 0.5997, 4: None}
MAP_AUPR     = {1: 0.0804, 2: 0.0019, 3: 0.0369, 4: 0.0190}
GCV_RF       = {1: 0.102,  2: 0.044,  3: 0.067,  4: 0.053}   # ridge factors


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Fig 1 — AUPR vs n/TF ratio
# ---------------------------------------------------------------------------

def fig_aupr_vs_ratio() -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    nets  = [1, 2, 3, 4]
    x     = [N_OVER_TF[n] for n in nets]
    y_g3  = [GENIE3_AUPR[n] for n in nets]
    y_gcv = [GCV_AUPR[n]    for n in nets]

    ax.plot(x, y_g3,  "o--", color=COLOURS["GENIE3"],    lw=2, ms=9,
            label="GENIE3 (Extra-Trees)")
    ax.plot(x, y_gcv, "s-",  color=COLOURS["GCV-Ridge"], lw=2, ms=9,
            label="GCV-Ridge (this work)")

    # Shade crossover region
    ax.axvspan(2.4, 4.1, alpha=0.07, color="grey", label="Transition zone")

    # Annotate network labels
    labels = {1: "net1\n(in silico)", 2: "net2\n(S. aureus)",
              3: "net3\n(E. coli)",   4: "net4\n(S. cerevisiae)"}
    offsets = {1: (0.06, 0.005), 2: (-0.35, 0.003),
               3: (0.06, 0.003), 4: (-0.35, -0.007)}
    for n in nets:
        dx, dy = offsets[n]
        ax.annotate(labels[n], xy=(N_OVER_TF[n], GCV_AUPR[n]),
                    xytext=(N_OVER_TF[n] + dx, GCV_AUPR[n] + dy),
                    fontsize=8.5, ha="left")

    ax.set_xlabel("n / D  (samples per TF)", fontsize=12)
    ax.set_ylabel("AUPR", fontsize=12)
    ax.set_title("GCV-Ridge vs GENIE3: performance crossover at n/D ≈ 3",
                 fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(1.2, 4.6)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    savefig(fig, "fig1_aupr_vs_ratio.png")


# ---------------------------------------------------------------------------
# Fig 2 — Ablation bar chart (net2 and net3, most informative)
# ---------------------------------------------------------------------------

def fig_ablation_bars() -> None:
    methods = ["GENIE3", "GCV-Ridge", "VI+KL", "MAP+W_init"]
    aupr = {
        "net2": [GENIE3_AUPR[2], GCV_AUPR[2], VI_AUPR[2],  MAP_AUPR[2]],
        "net3": [GENIE3_AUPR[3], GCV_AUPR[3], VI_AUPR[3],  MAP_AUPR[3]],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    net_titles = {
        "net2": "net2 — S. aureus\n(n/D = 1.62, n = 160)",
        "net3": "net3 — E. coli\n(n/D = 2.41, n = 805)",
    }
    x = np.arange(len(methods))
    bar_colours = [COLOURS[m] for m in methods]

    for ax, (net_key, vals) in zip(axes, aupr.items()):
        bars = ax.bar(x, vals, color=bar_colours, width=0.6, edgecolor="white", lw=0.8)
        ax.axhline(vals[0], color=COLOURS["GENIE3"], lw=1.2, ls="--", alpha=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0004,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel("AUPR", fontsize=11)
        ax.set_title(net_titles[net_key], fontsize=11)
        ax.set_ylim(bottom=0, top=max(vals) * 1.18)

    fig.suptitle("Ablation: AUPR by method on biological DREAM5 networks",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    savefig(fig, "fig2_ablation_bars.png")


# ---------------------------------------------------------------------------
# Fig 3 — GCV calibration curves for each network
# ---------------------------------------------------------------------------

def fig_gcv_curves() -> None:
    nets_to_run = [1, 2, 3, 4]
    fig, axes   = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat   = axes.ravel()

    net_info = {
        1: ("net1 (in silico)",       "1e-06", 0.102, 805),
        2: ("net2 (S. aureus)",       "6.25e-07", 0.044, 160),
        3: ("net3 (E. coli)",         "1.24e-07", 0.067, 805),
        4: ("net4 (S. cerevisiae)",   "1.87e-07", 0.053, 536),
    }

    for ax, nid in zip(axes_flat, nets_to_run):
        net = load_network(DATA_DIR, nid)

        expr     = net.expression.values.astype(np.float64)
        n        = expr.shape[0]
        gene_std = np.maximum(expr.std(0, keepdims=True), 1e-8)
        expr_std = (expr - expr.mean(0, keepdims=True)) / gene_std

        tf_idx = [i for i, g in enumerate(net.gene_ids) if g in set(net.tf_ids)]
        X = expr_std[:, tf_idx]
        Y = expr_std

        U, d, _ = np.linalg.svd(X, full_matrices=False)
        d2  = d ** 2
        UtY = U.T @ Y

        alpha_grid = np.logspace(-4, math.log10(1e4 * n), 80)
        gcv_scores = []
        for alpha in alpha_grid:
            trace_H   = (d2 / (d2 + alpha)).sum()
            shrink    = d2 / (d2 + alpha)
            Y_hat     = U @ (shrink[:, None] * UtY)
            rss       = ((Y - Y_hat) ** 2).sum()
            denom     = n * (1.0 - trace_H / n) ** 2
            gcv_scores.append(rss / denom)

        gcv_scores = np.array(gcv_scores)
        best_idx   = int(np.argmin(gcv_scores))
        alpha_star = alpha_grid[best_idx]
        rf_star    = alpha_star / n

        ax.semilogx(alpha_grid / n, gcv_scores / gcv_scores.max(),
                    color=COLOURS["GCV-Ridge"], lw=2)
        ax.axvline(rf_star, color="red", lw=1.5, ls="--",
                   label=f"α* = {alpha_star:.2g}  (rf = {rf_star:.3g})")
        ax.set_xlabel("ridge_factor  (α / n)", fontsize=10)
        ax.set_ylabel("Normalised GCV score", fontsize=10)
        title, _, _, _ = net_info[nid]
        ax.set_title(f"{title}\nn={net.n_samples}, D={net.n_tfs}, n/D={n/len(tf_idx):.2f}",
                     fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("GCV calibration curves — optimal ridge_factor per network",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    savefig(fig, "fig3_gcv_curves.png")


# ---------------------------------------------------------------------------
# Fig 4 — VI horseshoe lambda collapse (from saved checkpoint)
# ---------------------------------------------------------------------------

def fig_vi_lambda_collapse() -> None:
    import torch

    ckpt_path = BJEPA_RESULTS / "net1" / "stage2_best.pt"
    if not ckpt_path.exists():
        print(f"  Skipping Fig 4: {ckpt_path} not found.")
        return

    sd = torch.load(ckpt_path, map_location="cpu")
    # m_lambda is the log-mean of the local scale variational distribution
    m_lambda_key = next((k for k in sd if "m_lambda" in k), None)
    if m_lambda_key is None:
        print("  Skipping Fig 4: m_lambda not found in checkpoint.")
        return

    m_lambda = sd[m_lambda_key].numpy().ravel()
    lam_tilde = np.exp(m_lambda)
    kappa_vi  = 1.0 / (1.0 + lam_tilde ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: λ̃ distribution
    axes[0].hist(lam_tilde, bins=80, color=COLOURS["VI+KL"], alpha=0.85,
                 edgecolor="white")
    axes[0].axvline(1.0, color="red", lw=2, ls="--", label="λ̃ = 1 (degenerate)")
    axes[0].set_xlabel("λ̃  (local scale, variational mode)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"VI: λ̃ distribution — net1\n"
                      f"mean={lam_tilde.mean():.3f}  std={lam_tilde.std():.4f}",
                      fontsize=11)
    axes[0].legend(fontsize=10)

    # Right: κ distribution
    axes[1].hist(kappa_vi, bins=80, color=COLOURS["VI+KL"], alpha=0.85,
                 edgecolor="white")
    axes[1].axvline(0.49, color="red", lw=2, ls="--", label="κ ≈ 0.49 (collapsed)")
    axes[1].axvline(0.0,  color="green", lw=1.5, ls=":", label="κ ≈ 0 (ideal signal)")
    axes[1].axvline(1.0,  color="blue",  lw=1.5, ls=":", label="κ ≈ 1 (ideal noise)")
    axes[1].set_xlabel("κ = 1/(1 + λ̃²)  (shrinkage coefficient)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"VI: κ distribution — net1\n"
                      f"mean={kappa_vi.mean():.3f}  std={kappa_vi.std():.4f}  "
                      f"(expected bimodal spike at 0 and 1)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 1)

    fig.suptitle("Mean-field VI horseshoe failure: λ̃ → 1 for all edges (κ → 0.49)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    savefig(fig, "fig4_vi_lambda_collapse.png")


# ---------------------------------------------------------------------------
# Fig 5 — Full comparison heatmap (AUPR, all methods × all networks)
# ---------------------------------------------------------------------------

def fig_comparison_heatmap() -> None:
    methods  = ["GENIE3", "GCV-Ridge", "VI+KL", "MAP+W_init"]
    net_keys = ["net1", "net2", "net3", "net4"]
    net_names = ["net1\n(in silico)", "net2\n(S. aureus)",
                 "net3\n(E. coli)",   "net4\n(S. cerevisiae)"]

    data = np.array([
        [GENIE3_AUPR[1], GENIE3_AUPR[2], GENIE3_AUPR[3], GENIE3_AUPR[4]],
        [GCV_AUPR[1],    GCV_AUPR[2],    GCV_AUPR[3],    GCV_AUPR[4]],
        [VI_AUPR[1],     VI_AUPR[2],     VI_AUPR[3],     float("nan")],
        [MAP_AUPR[1],    MAP_AUPR[2],    MAP_AUPR[3],    MAP_AUPR[4]],
    ])

    # Normalise per-column (per network) so each column max=1
    col_max  = np.nanmax(data, axis=0, keepdims=True)
    data_norm = data / col_max

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    # Annotate cells with raw AUPR
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            norm_val = data_norm[i, j]
            text_col = "white" if norm_val > 0.6 else "black"
            label = f"{val:.4f}" if not np.isnan(val) else "—"
            star  = " ★" if (not np.isnan(val)) and val == col_max[0, j] else ""
            ax.text(j, i, label + star, ha="center", va="center",
                    fontsize=10, color=text_col, fontweight="bold" if star else "normal")

    ax.set_xticks(range(4))
    ax.set_xticklabels(net_names, fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    plt.colorbar(im, ax=ax, label="AUPR (normalised per network)", fraction=0.03)
    ax.set_title("AUPR comparison — all methods × all networks  (★ = best per network)",
                 fontsize=11)
    plt.tight_layout()
    savefig(fig, "fig5_comparison_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 6-9 — PyMC figures (copy from results/pymc_hs if available)
# ---------------------------------------------------------------------------

def copy_pymc_figures() -> None:
    if not PYMC_RESULTS.exists():
        print("  PyMC results not found. Run exp4 first:")
        print("    python experiments/exp4_pymc_horseshoe/run.py --network 2")
        return

    copied = 0
    for fig_path in PYMC_RESULTS.rglob("*.png"):
        dest = FIG_DIR / f"pymc_{fig_path.name}"
        shutil.copy2(fig_path, dest)
        print(f"  Copied: pymc_{fig_path.name}")
        copied += 1

    if copied == 0:
        print("  No PyMC figures found yet. Run exp4 first.")
    else:
        print(f"  {copied} PyMC figures copied to {FIG_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pymc", action="store_true",
                        help="Skip copying PyMC figures (if exp4 not yet run)")
    parser.add_argument("--only", type=str, default=None,
                        help="Generate only one figure: 1,2,3,4,5,pymc")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {FIG_DIR}/\n")

    only = args.only

    if only in (None, "1"):
        print("Fig 1 — AUPR vs n/TF ratio...")
        fig_aupr_vs_ratio()

    if only in (None, "2"):
        print("Fig 2 — Ablation bar chart...")
        fig_ablation_bars()

    if only in (None, "3"):
        print("Fig 3 — GCV calibration curves (loads all 4 networks)...")
        fig_gcv_curves()

    if only in (None, "4"):
        print("Fig 4 — VI lambda collapse (loads net1 stage2_best.pt)...")
        fig_vi_lambda_collapse()

    if only in (None, "5"):
        print("Fig 5 — Comparison heatmap...")
        fig_comparison_heatmap()

    if not args.no_pymc and only in (None, "pymc"):
        print("Figs 6-9 — PyMC NUTS figures...")
        copy_pymc_figures()

    print(f"\nDone. All figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
