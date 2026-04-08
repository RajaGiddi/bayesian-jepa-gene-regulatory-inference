"""3-way ensemble: GCV-Ridge + Per-gene GCV + B-JEPA.

Sweeps weights (α_gcv, α_pg, α_bjepa) on a simplex grid where
α_gcv + α_pg + α_bjepa = 1.0, and reports the best AUPR.

Usage:
    python experiments/ablations/three_way_ensemble.py
    python experiments/ablations/three_way_ensemble.py --network 2 --steps 20
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
from bjepa.models import BJEPAStage2
from bjepa.models.horseshoe import compute_tau0

# Import per_gene_gcv_scores directly by path (experiments/ is not a package)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "per_gene_gcv",
    ROOT / "experiments" / "ablations" / "per_gene_gcv.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
per_gene_gcv_scores = _mod.per_gene_gcv_scores

DATA_DIR  = ROOT / "data"
GCV_DIR   = ROOT / "results" / "analytical_hs" / "ols"
BJEPA_DIR = ROOT / "results" / "bjepa"


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["score"].rank(method="average") / len(df)
    return df


def load_gcv(nid: int) -> pd.DataFrame:
    df = pd.read_csv(GCV_DIR / f"net{nid}" / "edge_scores.csv")
    df["score"] = df["score"].abs()
    return rank_normalise(df)


def load_bjepa(nid: int, network, device="cpu") -> pd.DataFrame:
    ckpt = BJEPA_DIR / f"net{nid}" / "stage2_best.pt"
    tau0   = compute_tau0(p0=10.0, n_tfs=network.n_tfs, n_samples=network.n_samples)
    stage2 = BJEPAStage2(n_tfs=network.n_tfs, n_genes=network.n_genes,
                         tau0=tau0, kl_weight=1.0)
    stage2.load_state_dict(torch.load(ckpt, map_location=device))
    stage2.eval()
    with torch.no_grad():
        df = stage2.edge_scores(network.gene_ids, network.tf_ids)
    df["score"] = df["score"].abs()
    return rank_normalise(df)


def load_pg_gcv(nid: int, network) -> pd.DataFrame:
    cache = ROOT / "results" / "ablations" / "per_gene_gcv" / f"net{nid}" / "per_gene_gcv_scores.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = per_gene_gcv_scores(network, horseshoe=True)
    df["score"] = df["score"].abs()
    return rank_normalise(df)


def run(nid: int, steps: int) -> None:
    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return

    print(f"\nnet{nid} ({network.name})  n/D={network.n_samples/network.n_tfs:.2f}")

    gcv   = load_gcv(nid)
    pg    = load_pg_gcv(nid, network)
    bjepa = load_bjepa(nid, network)

    # Merge all three on (tf, target)
    merged = gcv.merge(pg,    on=["tf", "target"], suffixes=("_gcv", "_pg"))
    merged = merged.merge(bjepa, on=["tf", "target"])
    merged = merged.rename(columns={"score": "score_bjepa"})

    # Baselines
    for label, col in [("GCV", "score_gcv"), ("PG-GCV", "score_pg"), ("B-JEPA", "score_bjepa")]:
        tmp = merged[["tf", "target", col]].rename(columns={col: "score"})
        m   = evaluate_predictions(tmp, network.gold_standard)
        print(f"  {label:<10} AUPR={m['aupr']:.4f}")

    # Sweep simplex: α_gcv + α_pg + α_bjepa = 1
    # Use steps evenly spaced in [0, 1] for each weight
    best = {"aupr": -1.0, "w": None}
    alphas = np.linspace(0, 1, steps + 1)

    for a_gcv in alphas:
        for a_pg in alphas:
            a_bj = 1.0 - a_gcv - a_pg
            if a_bj < -1e-9:
                continue
            a_bj = max(0.0, a_bj)
            scores = (a_gcv * merged["score_gcv"] +
                      a_pg  * merged["score_pg"]  +
                      a_bj  * merged["score_bjepa"])
            tmp = merged[["tf", "target"]].copy()
            tmp["score"] = scores
            m = evaluate_predictions(tmp, network.gold_standard)
            if m["aupr"] > best["aupr"]:
                best = {"aupr": m["aupr"], "auroc": m["auroc"],
                        "w": (round(float(a_gcv), 3),
                              round(float(a_pg), 3),
                              round(float(a_bj), 3))}

    w = best["w"]
    print(f"  {'3-way best':<10} AUPR={best['aupr']:.4f}"
          f"  (GCV={w[0]}  PG={w[1]}  BJEPA={w[2]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--steps",   type=int, default=20,
                        help="Grid steps per axis (20 → 231 evaluations)")
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]
    for nid in nids:
        run(nid, args.steps)


if __name__ == "__main__":
    main()
