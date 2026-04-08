"""B-JEPA two-stage experiment runner.

Usage (from repo root, inside bayes_stats env):

    # Run all networks:
    python experiments/exp2_bjepa/run.py

    # Run a single network:
    python experiments/exp2_bjepa/run.py --network 3

    # Custom config:
    python experiments/exp2_bjepa/run.py --config configs/bjepa_default.yaml

Results written to results/bjepa/net{id}/.
Final table compares B-JEPA vs GENIE3.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import pandas as pd
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.models import BJEPAStage1, BJEPAStage2
from bjepa.training import BJEPATrainer, TrainerConfig

DATA_DIR       = ROOT / "data"
RESULTS_DIR    = ROOT / "results" / "bjepa"
GENIE3_SUMMARY = ROOT / "results" / "genie3" / "summary.json"
DEFAULT_CONFIG = ROOT / "configs" / "bjepa_default.yaml"


def load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def detect_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_network(nid: int, cfg: dict) -> dict:
    net = load_network(DATA_DIR, nid)
    if net.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    device      = detect_device(cfg["training"]["device"])
    d_latent    = cfg["model"]["d_latent"]
    net_results = RESULTS_DIR / f"net{nid}"

    stage1 = BJEPAStage1.from_network(
        net,
        d_latent         = d_latent,
        encoder_hidden   = cfg["model"]["encoder_hidden"],
        encoder_dropout  = cfg["model"]["encoder_dropout"],
        ema_momentum     = cfg["model"]["ema_momentum"],
        kl_weight        = cfg["model"].get("stage1_kl_weight", 1.0),
        predictor_hidden = cfg["model"].get("predictor_hidden", 256),
    )
    stage2 = BJEPAStage2.from_network(
        net,
        p0        = cfg["model"]["p0"],
        kl_weight = 1.0,
    )

    trainer_cfg = TrainerConfig(
        stage1_epochs       = cfg["training"]["stage1_epochs"],
        stage1_lr           = cfg["training"]["lr"],
        stage1_weight_decay = cfg["training"]["weight_decay"],
        stage2_epochs       = cfg["training"]["stage2_epochs"],
        stage2_lr           = cfg["training"]["stage2_lr"],
        kl_warmup_epochs    = cfg["training"]["kl_warmup_epochs"],
        kl_weight_max       = cfg["training"]["kl_weight_max"],
        kl_mc_samples       = cfg["training"]["kl_mc_samples"],
        eval_every          = cfg["training"]["eval_every"],
        results_dir         = str(net_results),
        device              = device,
    )

    t0      = time.perf_counter()
    trainer = BJEPATrainer(stage1, stage2, net, trainer_cfg)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    if history.aupr:
        best_idx   = int(max(range(len(history.aupr)), key=lambda i: history.aupr[i]))
        best_auroc = history.auroc[best_idx]
        best_aupr  = history.aupr[best_idx]
        best_epoch = history.eval_epochs[best_idx]
    else:
        best_auroc = best_aupr = float("nan")
        best_epoch = -1

    result = {
        "network":          f"net{nid} ({net.name})",
        "auroc":            best_auroc,
        "aupr":             best_aupr,
        "best_s2_epoch":    best_epoch,
        "elapsed_s":        round(elapsed, 1),
        "n_samples":        net.n_samples,
        "n_genes":          net.n_genes,
        "n_tfs":            net.n_tfs,
        "n_positive_edges": net.n_positive_edges,
        "device":           device,
    }
    with open(net_results / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def print_comparison(bjepa_results: list[dict]) -> None:
    if not bjepa_results:
        return
    bjepa_df = pd.DataFrame(bjepa_results).set_index("network")

    if GENIE3_SUMMARY.exists():
        with open(GENIE3_SUMMARY) as f:
            g3 = pd.DataFrame(json.load(f)).set_index("network")
        common = bjepa_df.index.intersection(g3.index)
        if len(common):
            cmp = pd.DataFrame({
                "GENIE3 AUROC": g3.loc[common, "auroc"],
                "GENIE3 AUPR":  g3.loc[common, "aupr"],
                "B-JEPA AUROC": bjepa_df.loc[common, "auroc"],
                "B-JEPA AUPR":  bjepa_df.loc[common, "aupr"],
                "ΔAUROC":       bjepa_df.loc[common, "auroc"] - g3.loc[common, "auroc"],
                "ΔAUPR":        bjepa_df.loc[common, "aupr"]  - g3.loc[common, "aupr"],
            })
            print("\n\nB-JEPA vs GENIE3:")
            print(cmp.to_string(float_format="{:.4f}".format))
            return

    print("\n\nB-JEPA results:")
    print(bjepa_df[["auroc", "aupr", "best_s2_epoch", "elapsed_s"]].to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--config",  type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    cfg  = load_config(pathlib.Path(args.config))
    nids = [args.network] if args.network else cfg["data"]["network_ids"]

    all_results = []
    for nid in nids:
        result = run_network(nid, cfg)
        if result:
            all_results.append(result)

    if all_results:
        summary_df = pd.DataFrame(all_results).set_index("network")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(RESULTS_DIR / "summary.csv")
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print_comparison(all_results)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
