"""GENIE3 baseline — run across all DREAM5 networks with a gold standard.

Usage (from repo root, inside bayes_stats env):

    # Run everything:
    python experiments/exp1_genie3_baseline/run.py

    # Run a single network:
    python experiments/exp1_genie3_baseline/run.py --network 3

    # Re-evaluate from saved predictions (no re-training):
    python experiments/exp1_genie3_baseline/run.py --reeval

Results are written to results/genie3/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.baselines import GENIE3
from bjepa.eval import evaluate_predictions

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "genie3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GENIE3_KWARGS = dict(n_estimators=1000, max_features="sqrt", n_jobs=-1, random_state=42)


def run_network(nid: int) -> dict:
    net = load_network(DATA_DIR, nid)
    if net.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    print(f"\n{'='*60}")
    print(f"net{nid} ({net.name}): {net.n_samples} samples, {net.n_genes} genes, {net.n_tfs} TFs")

    model = GENIE3(**GENIE3_KWARGS)
    t0 = time.perf_counter()
    predictions = model.fit_predict(net)
    elapsed = time.perf_counter() - t0

    pred_path = RESULTS_DIR / f"net{nid}_predictions.tsv"
    predictions.to_csv(pred_path, sep="\t", index=False)

    return _evaluate_and_print(nid, net, predictions, elapsed)


def reeval_network(nid: int) -> dict:
    """Re-evaluate from saved predictions (no GENIE3 re-run)."""
    pred_path = RESULTS_DIR / f"net{nid}_predictions.tsv"
    if not pred_path.exists():
        print(f"net{nid}: no saved predictions at {pred_path}, skipping.")
        return {}

    net = load_network(DATA_DIR, nid)
    if net.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    print(f"\n{'='*60}")
    print(f"net{nid} ({net.name}): re-evaluating from {pred_path.name}")
    predictions = pd.read_csv(pred_path, sep="\t")
    return _evaluate_and_print(nid, net, predictions, elapsed=None)


def _evaluate_and_print(nid, net, predictions, elapsed) -> dict:
    metrics = evaluate_predictions(predictions, net.gold_standard)
    metrics["network"] = f"net{nid} ({net.name})"
    metrics["elapsed_s"] = round(elapsed, 1) if elapsed is not None else None
    metrics["n_samples"] = net.n_samples
    metrics["n_genes"] = net.n_genes
    metrics["n_tfs"] = net.n_tfs
    metrics["n_positive_edges"] = net.n_positive_edges

    auroc_str = f"{metrics['auroc']:.4f}" if metrics['auroc'] == metrics['auroc'] else "nan"
    elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "cached"
    print(f"  AUROC = {auroc_str}  |  AUPR = {metrics['aupr']:.4f}  |  {elapsed_str}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4],
                        help="Run a single network (default: all)")
    parser.add_argument("--reeval", action="store_true",
                        help="Re-evaluate from saved predictions without re-running GENIE3")
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]
    fn = reeval_network if args.reeval else run_network

    all_metrics = [m for nid in nids if (m := fn(nid))]

    df = pd.DataFrame(all_metrics).set_index("network")
    print("\n\nGENIE3 baseline results:")
    print(df[["auroc", "aupr", "elapsed_s"]].to_string())

    df.to_csv(RESULTS_DIR / "summary.csv")
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
