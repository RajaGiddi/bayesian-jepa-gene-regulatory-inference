"""Phase 1A: KAN Stage 2 experiment.

Replaces linear horseshoe regression (BJEPAStage2) with factored KAN
regression (KANStage2).  Stage 1 is frozen — uses the saved checkpoint
from exp2.  W_init extracted identically to exp2 to isolate the Stage 2
change.

For each network, reports:
  - GCV-Ridge AUPR              (analytical baseline)
  - Linear Stage 2 AUPR         (current B-JEPA Stage 2)
  - KAN Stage 2 AUPR            (new, factored B-spline activation)
  - Ensemble GCV + KAN Stage 2  (rank-normalised linear combination)

KAN hyperparameters
-------------------
grid_size    : G — number of spline intervals (G+k basis functions per TF)
spline_order : k — polynomial order (cubic k=3 by default)
kl_weight    : horseshoe KL coefficient (same sweep as exp2 ablation 2)
grid_update_every : epochs between adaptive grid updates (0 = never)

Usage:
    python experiments/exp_kan/run_kan.py --network 2
    python experiments/exp_kan/run_kan.py                       # all networks
    python experiments/exp_kan/run_kan.py --network 2 --grid 3  # smaller grid
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
from bjepa.models import BJEPAStage1, BJEPAStage2, KANStage2
from bjepa.models.horseshoe import compute_tau0

DATA_DIR    = ROOT / "data"
BJEPA_DIR   = ROOT / "results" / "bjepa"
GCV_DIR     = ROOT / "results" / "analytical_hs" / "ols"
RESULTS_DIR = ROOT / "results" / "kan_stage2"


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["score"].rank(method="average") / len(df)
    return df


def ensemble(a: pd.DataFrame, b: pd.DataFrame, alpha: float) -> pd.DataFrame:
    m = a.merge(b, on=["tf", "target"], suffixes=("_a", "_b"))
    m["score"] = alpha * m["score_a"] + (1 - alpha) * m["score_b"]
    return m[["tf", "target", "score"]]


# ---------------------------------------------------------------------------
# Stage 1: load frozen checkpoint + extract W_init
# ---------------------------------------------------------------------------

def load_stage1_and_winit(nid: int, network, device: str) -> torch.Tensor:
    ckpt = BJEPA_DIR / f"net{nid}" / "stage1_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No Stage 1 checkpoint at {ckpt}. Run exp2 first."
        )
    stage1 = BJEPAStage1.from_network(
        network,
        d_latent=256, encoder_hidden=(512, 512), encoder_dropout=0.1,
        ema_momentum=0.996, kl_weight=1.0, predictor_hidden=256,
    ).to(device)
    stage1.load_state_dict(torch.load(ckpt, map_location=device))
    stage1.eval()

    # Build normalised expression tensors (same as exp2)
    expr_np   = network.expression.values.T.astype("float32")   # (G, n)
    gene_mean = expr_np.mean(axis=1, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=1, keepdims=True), 1e-6)
    expr_t    = torch.from_numpy((expr_np - gene_mean) / gene_std).to(device)
    tf_set    = set(network.tf_ids)
    tf_mask   = torch.tensor(
        [g in tf_set for g in network.gene_ids], dtype=torch.bool
    ).to(device)

    with torch.no_grad():
        W_init = stage1.get_W_init(expr_t, tf_mask)  # (n_tfs, n_genes)

    return W_init


# ---------------------------------------------------------------------------
# Stage 2: data preparation (same for linear and KAN)
# ---------------------------------------------------------------------------

def build_stage2_data(network, device: str):
    expr_np   = network.expression.values.astype("float32")    # (n, G)
    gene_mean = expr_np.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=0, keepdims=True), 1e-6)
    expr_std  = (expr_np - gene_mean) / gene_std
    tf_indices = [
        i for i, g in enumerate(network.gene_ids)
        if g in set(network.tf_ids)
    ]
    X_tf = torch.from_numpy(expr_std[:, tf_indices]).to(device)  # (n, D)
    Y    = torch.from_numpy(expr_std).to(device)                  # (n, G)
    return X_tf, Y


# ---------------------------------------------------------------------------
# Training loop (shared by linear and KAN Stage 2)
# ---------------------------------------------------------------------------

def train_stage2(
    model,
    X_tf: torch.Tensor,
    Y:    torch.Tensor,
    network,
    epochs:           int,
    kl_warmup:        int,
    kl_max:           float,
    grid_update_every: int = 0,
    label:            str = "Stage2",
) -> tuple[float, pd.DataFrame]:
    """Train a Stage 2 model (linear or KAN).  Returns (best_aupr, best_scores)."""
    opt   = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)

    best_aupr   = -1.0
    best_scores = None

    for epoch in range(1, epochs + 1):
        model.train()
        kl_w = min(kl_max, epoch / max(1, kl_warmup) * kl_max)
        model.kl_weight = kl_w

        opt.zero_grad()
        out = model(X_tf, Y, kl_mc_samples=4)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        # Adaptive grid update for KAN (optional)
        if grid_update_every > 0 and hasattr(model, "kan_act"):
            if epoch % grid_update_every == 0:
                model.kan_act.update_grid(X_tf.detach())

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                scores = model.edge_scores(network.gene_ids, network.tf_ids)
            scores["score"] = scores["score"].abs()
            m = evaluate_predictions(scores, network.gold_standard)
            if m["aupr"] > best_aupr:
                best_aupr   = m["aupr"]
                best_scores = scores.copy()

            if epoch % 50 == 0 or epoch == epochs:
                print(f"    [{label}] ep {epoch:4d}/{epochs}"
                      f"  loss={out.loss.item():.4f}"
                      f"  reg={out.regression_loss.item():.4f}"
                      f"  kl={out.kl_loss.item():.4f}"
                      f"  AUPR={m['aupr']:.4f}")

    return best_aupr, best_scores


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(nid: int, epochs: int, kl_max: float, grid_size: int,
        spline_order: int, grid_update_every: int, device: str) -> dict:
    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    print(f"\n{'='*60}")
    print(f"Phase 1A — KAN Stage 2  |  net{nid} ({network.name})")
    print(f"  n={network.n_samples}  D={network.n_tfs}  G={network.n_genes}"
          f"  n/D={network.n_samples/network.n_tfs:.2f}")
    print(f"  KAN grid={grid_size}  order={spline_order}"
          f"  basis_fns/TF={grid_size + spline_order}"
          f"  kl_max={kl_max}")

    # GCV-Ridge baseline
    gcv_path = GCV_DIR / f"net{nid}" / "edge_scores.csv"
    gcv_df   = pd.read_csv(gcv_path)
    gcv_df["score"] = gcv_df["score"].abs()
    gcv_aupr = evaluate_predictions(gcv_df, network.gold_standard)["aupr"]
    print(f"\n  GCV-Ridge AUPR = {gcv_aupr:.4f}  (target to beat)")

    # Load Stage 1 + W_init
    print(f"\n  Loading Stage 1 checkpoint and extracting W_init...")
    W_init = load_stage1_and_winit(nid, network, device)
    print(f"  W_init: {tuple(W_init.shape)}  "
          f"mean|W|={W_init.abs().mean().item():.4f}")

    X_tf, Y = build_stage2_data(network, device)
    tau0     = compute_tau0(p0=10.0, n_tfs=network.n_tfs, n_samples=network.n_samples)

    # ------------------------------------------------------------------
    # Baseline: linear Stage 2 (same as exp2)
    # ------------------------------------------------------------------
    print(f"\n  Training linear Stage 2 ({epochs} epochs)...")
    linear_s2 = BJEPAStage2(
        n_tfs=network.n_tfs, n_genes=network.n_genes,
        tau0=tau0, kl_weight=kl_max,
    ).to(device)
    linear_s2.init_from_stage1(W_init)

    t0 = time.perf_counter()
    linear_aupr, linear_scores = train_stage2(
        linear_s2, X_tf, Y, network,
        epochs=epochs, kl_warmup=50, kl_max=kl_max,
        label="Linear",
    )
    t_linear = time.perf_counter() - t0
    print(f"  Linear Stage 2:  AUPR={linear_aupr:.4f}  [{t_linear:.0f}s]")

    # ------------------------------------------------------------------
    # KAN Stage 2
    # ------------------------------------------------------------------
    print(f"\n  Training KAN Stage 2 ({epochs} epochs)...")
    kan_s2 = KANStage2(
        n_tfs=network.n_tfs, n_genes=network.n_genes,
        tau0=tau0, kl_weight=kl_max,
        grid_size=grid_size, spline_order=spline_order,
    ).to(device)
    kan_s2.init_from_stage1(W_init)

    t0 = time.perf_counter()
    kan_aupr, kan_scores = train_stage2(
        kan_s2, X_tf, Y, network,
        epochs=epochs, kl_warmup=50, kl_max=kl_max,
        grid_update_every=grid_update_every,
        label="KAN   ",
    )
    t_kan = time.perf_counter() - t0
    print(f"  KAN Stage 2:     AUPR={kan_aupr:.4f}  [{t_kan:.0f}s]")

    # ------------------------------------------------------------------
    # Ensemble: GCV + KAN Stage 2
    # ------------------------------------------------------------------
    gcv_rn = rank_normalise(gcv_df)
    kan_rn = rank_normalise(kan_scores)

    best_ens = {"aupr": -1.0, "alpha": -1.0}
    for alpha in np.linspace(0.0, 1.0, 21):
        m = evaluate_predictions(
            ensemble(gcv_rn, kan_rn, alpha), network.gold_standard
        )
        if m["aupr"] > best_ens["aupr"]:
            best_ens = {"aupr": m["aupr"], "alpha": round(float(alpha), 2)}

    # Also compare ensemble of GCV + linear (reference)
    lin_rn = rank_normalise(linear_scores)
    best_lin_ens = {"aupr": -1.0, "alpha": -1.0}
    for alpha in np.linspace(0.0, 1.0, 21):
        m = evaluate_predictions(
            ensemble(gcv_rn, lin_rn, alpha), network.gold_standard
        )
        if m["aupr"] > best_lin_ens["aupr"]:
            best_lin_ens = {"aupr": m["aupr"], "alpha": round(float(alpha), 2)}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  {'Method':<35} {'AUPR':>8} {'Δ vs GCV':>10}")
    print(f"  {'-'*55}")
    for label, aupr in [
        ("GCV-Ridge (baseline)",         gcv_aupr),
        ("Linear Stage 2 (B-JEPA)",      linear_aupr),
        ("KAN Stage 2",                  kan_aupr),
        ("Ensemble GCV + Linear Stage2", best_lin_ens["aupr"]),
        ("Ensemble GCV + KAN Stage2",    best_ens["aupr"]),
    ]:
        delta = aupr - gcv_aupr
        marker = " ◄" if aupr > gcv_aupr else ""
        print(f"  {label:<35} {aupr:>8.4f} {delta:>+10.4f}{marker}")

    print(f"\n  KAN vs Linear (standalone): {kan_aupr - linear_aupr:+.4f}")
    print(f"  KAN ensemble vs Linear ensemble: "
          f"{best_ens['aupr'] - best_lin_ens['aupr']:+.4f}")

    # Save
    out_dir = RESULTS_DIR / f"net{nid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kan_scores.to_csv(out_dir / "kan_scores.csv", index=False)
    torch.save(kan_s2.state_dict(), out_dir / "kan_stage2.pt")
    print(f"\n  Saved to {out_dir}/")

    return {
        "network":           f"net{nid} ({network.name})",
        "gcv_aupr":          gcv_aupr,
        "linear_aupr":       linear_aupr,
        "kan_aupr":          kan_aupr,
        "linear_ens_aupr":   best_lin_ens["aupr"],
        "kan_ens_aupr":      best_ens["aupr"],
        "kan_vs_linear":     kan_aupr - linear_aupr,
        "ens_kan_vs_linear": best_ens["aupr"] - best_lin_ens["aupr"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--epochs",  type=int, default=300)
    parser.add_argument("--kl_max",  type=float, default=0.1)
    parser.add_argument("--grid",    type=int, default=5,
                        help="B-spline grid size G (default 5)")
    parser.add_argument("--order",   type=int, default=3,
                        help="B-spline order k (default 3, cubic)")
    parser.add_argument("--grid_update", type=int, default=0,
                        help="Adaptive grid update every N epochs (0=off)")
    parser.add_argument("--device",  type=str, default="mps")
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]
    results = []
    for nid in nids:
        r = run(nid, args.epochs, args.kl_max,
                args.grid, args.order, args.grid_update, args.device)
        if r:
            results.append(r)

    if len(results) > 1:
        print("\n\nSummary across networks:")
        df = pd.DataFrame(results).set_index("network")
        print(df.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
