"""Ablation 2: kl_weight sweep in Stage 2.

Reuses the saved Stage 1 checkpoint and sweeps kl_weight_max over Stage 2
to find the optimal KL regularisation strength.  The VI horseshoe implicitly
acts as ridge; kl_weight controls the effective ridge factor.  GCV finds the
optimal ridge analytically — this sweep checks whether VI can match it.

Usage (from repo root):
    python experiments/ablations/kl_weight_sweep.py
    python experiments/ablations/kl_weight_sweep.py --network 2
    python experiments/ablations/kl_weight_sweep.py --network 2 --epochs 100
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bjepa.data import load_network
from bjepa.eval.metrics import evaluate_predictions
from bjepa.models import BJEPAStage1, BJEPAStage2
from bjepa.models.horseshoe import compute_tau0

DATA_DIR  = ROOT / "data"
BJEPA_DIR = ROOT / "results" / "bjepa"

KL_WEIGHTS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]


def load_stage1(nid: int, network, device: str, cfg: dict) -> BJEPAStage1:
    ckpt = BJEPA_DIR / f"net{nid}" / "stage1_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No Stage 1 checkpoint at {ckpt}. Run exp2 first.")
    stage1 = BJEPAStage1.from_network(
        network,
        d_latent         = cfg["d_latent"],
        encoder_hidden   = cfg["encoder_hidden"],
        encoder_dropout  = cfg["encoder_dropout"],
        ema_momentum     = cfg["ema_momentum"],
        predictor_hidden = cfg["predictor_hidden"],
    )
    stage1.load_state_dict(torch.load(ckpt, map_location=device))
    stage1.eval()
    return stage1.to(device)


def run_stage2(
    stage2: BJEPAStage2,
    X_tf: torch.Tensor,
    Y: torch.Tensor,
    network,
    kl_weight_max: float,
    epochs: int,
    kl_warmup_epochs: int,
    device: str,
) -> float:
    """Train Stage 2 with given kl_weight_max. Returns best AUPR."""
    opt   = AdamW(stage2.parameters(), lr=1e-3, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)

    best_aupr = -1.0

    for epoch in range(1, epochs + 1):
        stage2.train()
        kl_weight = min(kl_weight_max,
                        epoch / max(1, kl_warmup_epochs) * kl_weight_max)
        stage2.kl_weight = kl_weight

        opt.zero_grad()
        out = stage2(X_tf, Y, kl_mc_samples=4)
        out.loss.backward()
        nn.utils.clip_grad_norm_(stage2.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            stage2.eval()
            with torch.no_grad():
                scores_df = stage2.edge_scores(network.gene_ids, network.tf_ids)
            scores_df["score"] = scores_df["score"].abs()
            m = evaluate_predictions(scores_df, network.gold_standard)
            if m["aupr"] > best_aupr:
                best_aupr = m["aupr"]

    return best_aupr


def run(nid: int, epochs: int, device: str) -> None:
    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return

    print(f"\nnet{nid} ({network.name})  |  {network.n_samples} samples, "
          f"{network.n_tfs} TFs, {network.n_genes} genes")
    print(f"  Stage 2 epochs: {epochs} per kl_weight value")

    # Model config (must match what exp2 used)
    cfg = dict(d_latent=256, encoder_hidden=(512, 512), encoder_dropout=0.1,
               ema_momentum=0.996, predictor_hidden=256)

    # Load frozen Stage 1
    stage1 = load_stage1(nid, network, device, cfg)

    # Build expression matrices for Stage 2
    expr_np   = network.expression.values.astype("float32")
    gene_mean = expr_np.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=0, keepdims=True), 1e-6)
    expr_std  = (expr_np - gene_mean) / gene_std
    tf_indices = [i for i, g in enumerate(network.gene_ids)
                  if g in set(network.tf_ids)]
    X_tf = torch.from_numpy(expr_std[:, tf_indices]).to(device)
    Y    = torch.from_numpy(expr_std).to(device)

    # Derive W_init once (reused for all kl_weight values)
    expr_t  = torch.from_numpy(
        (network.expression.values.T.astype("float32") -
         network.expression.values.T.astype("float32").mean(axis=1, keepdims=True)) /
        np.maximum(network.expression.values.T.astype("float32").std(axis=1, keepdims=True), 1e-6)
    ).to(device)
    tf_set  = set(network.tf_ids)
    tf_mask = torch.tensor([g in tf_set for g in network.gene_ids],
                           dtype=torch.bool).to(device)

    with torch.no_grad():
        W_init = stage1.get_W_init(expr_t, tf_mask)

    tau0 = compute_tau0(p0=10.0, n_tfs=network.n_tfs, n_samples=network.n_samples)

    # GCV-Ridge baseline
    gcv_path = ROOT / "results" / "analytical_hs" / "ols" / f"net{nid}" / "edge_scores.csv"
    if gcv_path.exists():
        gcv_df = pd.read_csv(gcv_path)
        gcv_df["score"] = gcv_df["score"].abs()
        gcv_aupr = evaluate_predictions(gcv_df, network.gold_standard)["aupr"]
    else:
        gcv_aupr = float("nan")

    print(f"\n  GCV-Ridge AUPR = {gcv_aupr:.4f}  (target to beat)")
    print(f"\n  {'kl_weight_max':<16} {'best AUPR':>10} {'vs GCV':>10}  {'time':>6}")
    print(f"  {'-'*46}")

    results = []
    for kl_w in KL_WEIGHTS:
        # Fresh Stage 2 for each kl_weight
        stage2 = BJEPAStage2(
            n_tfs=network.n_tfs, n_genes=network.n_genes,
            tau0=tau0, kl_weight=kl_w,
        ).to(device)
        stage2.init_from_stage1(W_init)

        t0 = time.perf_counter()
        best_aupr = run_stage2(
            stage2, X_tf, Y, network,
            kl_weight_max=kl_w,
            epochs=epochs,
            kl_warmup_epochs=50,
            device=device,
        )
        elapsed = time.perf_counter() - t0

        delta = best_aupr - gcv_aupr
        marker = " ◄" if best_aupr > gcv_aupr else ""
        print(f"  {kl_w:<16.3f} {best_aupr:>10.4f} {delta:>+10.4f}  {elapsed:>5.0f}s{marker}")
        results.append({"kl_weight_max": kl_w, "aupr": best_aupr})

    best = max(results, key=lambda r: r["aupr"])
    print(f"\n  Best: kl_weight_max={best['kl_weight_max']}  AUPR={best['aupr']:.4f}")
    print(f"  GCV-Ridge:          AUPR={gcv_aupr:.4f}")
    print(f"  Default (0.1):      AUPR=0.0164  [from full exp2 run]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--device",  type=str, default="mps")
    args = parser.parse_args()
    run(args.network, args.epochs, args.device)


if __name__ == "__main__":
    main()
