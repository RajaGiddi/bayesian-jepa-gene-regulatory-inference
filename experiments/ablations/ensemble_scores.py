"""Ablation 1: Score ensemble — GCV-Ridge + B-JEPA.

Tests whether linearly combining GCV-Ridge and B-JEPA Stage 2 edge scores
exceeds either method alone.  Sweeps alpha from 0 (pure B-JEPA) to 1
(pure GCV-Ridge) and reports AUPR at each step.

Usage (from repo root):
    python experiments/ablations/ensemble_scores.py
    python experiments/ablations/ensemble_scores.py --network 2
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.eval.metrics import evaluate_predictions
from bjepa.models import BJEPAStage1, BJEPAStage2
from bjepa.models.horseshoe import compute_tau0

DATA_DIR    = ROOT / "data"
BJEPA_DIR   = ROOT / "results" / "bjepa"
GCV_DIR     = ROOT / "results" / "analytical_hs"


def load_gcv_scores(nid: int) -> pd.DataFrame:
    path = GCV_DIR / "ols" / f"net{nid}" / "edge_scores.csv"
    df = pd.read_csv(path)
    # score column is signed; use absolute value for ranking (same as exp3)
    df["score"] = df["score"].abs()
    return df[["tf", "target", "score"]]


def load_bjepa_scores(nid: int, network, device: str) -> pd.DataFrame:
    ckpt = BJEPA_DIR / f"net{nid}" / "stage2_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No B-JEPA checkpoint at {ckpt}. Run exp2 first.")

    tau0 = compute_tau0(p0=10.0, n_tfs=network.n_tfs, n_samples=network.n_samples)
    stage2 = BJEPAStage2(n_tfs=network.n_tfs, n_genes=network.n_genes,
                         tau0=tau0, kl_weight=1.0)
    stage2.load_state_dict(torch.load(ckpt, map_location=device))
    stage2.eval()

    with torch.no_grad():
        df = stage2.edge_scores(network.gene_ids, network.tf_ids)

    df["score"] = df["score"].abs()
    return df[["tf", "target", "score"]]


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Replace scores with fractional ranks in [0, 1]. Higher score → higher rank."""
    df = df.copy()
    df["score"] = df["score"].rank(method="average") / len(df)
    return df


def ensemble(gcv: pd.DataFrame, bjepa: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """alpha=1 → pure GCV-Ridge, alpha=0 → pure B-JEPA."""
    merged = gcv.merge(bjepa, on=["tf", "target"], suffixes=("_gcv", "_bjepa"))
    merged["score"] = alpha * merged["score_gcv"] + (1 - alpha) * merged["score_bjepa"]
    return merged[["tf", "target", "score"]]


def run(nid: int, device: str = "cpu") -> None:
    net = load_network(DATA_DIR, nid)
    if net.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return

    print(f"\nnet{nid} ({net.name})  |  {net.n_samples} samples, {net.n_tfs} TFs, {net.n_genes} genes")
    print(f"  Positive edges: {net.n_positive_edges} / {net.n_tfs * net.n_genes}")

    # Load and rank-normalise both score sets
    gcv   = rank_normalise(load_gcv_scores(nid))
    bjepa = rank_normalise(load_bjepa_scores(nid, net, device))

    # Baseline metrics
    gcv_metrics   = evaluate_predictions(gcv,   net.gold_standard)
    bjepa_metrics = evaluate_predictions(bjepa, net.gold_standard)
    print(f"\n  {'Method':<20} {'AUROC':>7} {'AUPR':>8}")
    print(f"  {'GCV-Ridge':<20} {gcv_metrics['auroc']:>7.4f} {gcv_metrics['aupr']:>8.4f}")
    print(f"  {'B-JEPA':<20} {bjepa_metrics['auroc']:>7.4f} {bjepa_metrics['aupr']:>8.4f}")
    print(f"  {'-'*38}")

    # Sweep alpha
    alphas = np.linspace(0.0, 1.0, 21)   # 0.00, 0.05, ..., 1.00
    best_aupr  = -1.0
    best_alpha = -1.0
    results = []

    for alpha in alphas:
        scores = ensemble(gcv, bjepa, alpha)
        m = evaluate_predictions(scores, net.gold_standard)
        results.append({"alpha": round(float(alpha), 2),
                        "auroc": m["auroc"], "aupr": m["aupr"]})
        if m["aupr"] > best_aupr:
            best_aupr  = m["aupr"]
            best_alpha = alpha

    # Print sweep
    print(f"  {'alpha (GCV weight)':<22} {'AUROC':>7} {'AUPR':>8}")
    for r in results:
        marker = " ◄ best" if abs(r["alpha"] - best_alpha) < 1e-6 else ""
        print(f"  {r['alpha']:<22.2f} {r['auroc']:>7.4f} {r['aupr']:>8.4f}{marker}")

    print(f"\n  Best ensemble AUPR = {best_aupr:.4f}  at alpha={best_alpha:.2f}")
    print(f"  vs GCV-Ridge       = {gcv_metrics['aupr']:.4f}")
    print(f"  vs B-JEPA          = {bjepa_metrics['aupr']:.4f}")
    delta = best_aupr - gcv_metrics["aupr"]
    print(f"  Δ vs GCV-Ridge     = {delta:+.4f}  ({'beats' if delta > 0 else 'misses'} target 0.018)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--device",  type=str, default="cpu")
    args = parser.parse_args()
    run(args.network, args.device)


if __name__ == "__main__":
    main()
