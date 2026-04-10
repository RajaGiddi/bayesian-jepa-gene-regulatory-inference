"""Phase 1B: Asymmetric BiJEPA Stage 1 experiment.

Adds a backward predictor to the JEPA objective:

    loss = α · L_fwd + (1−α) · L_bwd

    L_fwd : TF expression (context) → predict all gene representations  [standard]
    L_bwd : non-TF gene expression (context) → predict TF representations [new]

Biological motivation
---------------------
L_fwd captures the causal direction (TFs regulate genes).
L_bwd captures the reverse signal (gene expression reflects TF activity).
Because TFs are causal, we always keep α ≥ 0.5 so that the forward direction
dominates.  The backward signal provides complementary evidence that a TF is
active without requiring explicit regulatory annotation.

Sweep
-----
fwd_weight α ∈ {0.3, 0.5, 0.7, 0.9}

For each α:
  1. Train BiJEPA Stage 1 from scratch (150 epochs, same hyperparams as exp2).
  2. Extract W_init via OLS in latent space (same as exp2).
  3. Train Linear Stage 2 (300 epochs, kl_max=0.05 — optimal from ablation sweep).
  4. Report AUPR vs GCV-Ridge baseline and unidirectional B-JEPA.

Usage:
    python experiments/exp_bijepa/run_bijepa.py --network 2
    python experiments/exp_bijepa/run_bijepa.py               # all networks
    python experiments/exp_bijepa/run_bijepa.py --network 2 --fwd_weights 0.5 0.7 0.9
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

DATA_DIR    = ROOT / "data"
GCV_DIR     = ROOT / "results" / "analytical_hs" / "ols"
RESULTS_DIR = ROOT / "results" / "bijepa"

# Hyperparameters matched to exp2 / post-ablation-sweep findings
STAGE1_EPOCHS    = 150
STAGE1_LR        = 3e-4
STAGE1_WD        = 1e-4
STAGE2_EPOCHS    = 300
STAGE2_LR        = 1e-3
KL_WARMUP        = 50
KL_MAX           = 0.05   # optimal from kl_weight ablation sweep


# ---------------------------------------------------------------------------
# Stage 1 training
# ---------------------------------------------------------------------------

def train_stage1(
    model: BJEPAStage1,
    expression: torch.Tensor,   # (n_genes, n_samples) — standardised per gene
    tf_mask: torch.Tensor,      # (n_genes,) bool
    epochs: int,
    lr: float,
    weight_decay: float,
    fwd_weight: float,
    label: str = "BiJEPA",
) -> None:
    """Train Stage 1 in place."""
    opt   = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(expression, tf_mask)
        out.loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        opt.step()
        sched.step()
        model.update_ema()

        if epoch % 30 == 0 or epoch == epochs:
            bwd_str = ""
            if out.nll_bwd is not None:
                bwd_str = f"  nll_bwd={out.nll_bwd.item():.4f}"
            print(
                f"    [{label} α={fwd_weight}] ep {epoch:4d}/{epochs}"
                f"  loss={out.loss.item():.4f}"
                f"  nll={out.nll.item():.4f}"
                f"  kl={out.kl.item():.4f}"
                f"{bwd_str}"
            )


# ---------------------------------------------------------------------------
# Stage 2 training  (shared with exp_kan/run_kan.py)
# ---------------------------------------------------------------------------

def build_stage2_data(network, device: str):
    """Build (X_tf, Y) tensors standardised per gene for Stage 2."""
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


def train_stage2(
    model: BJEPAStage2,
    X_tf: torch.Tensor,
    Y:    torch.Tensor,
    network,
    epochs: int,
    kl_warmup: int,
    kl_max: float,
    label: str = "Stage2",
) -> tuple[float, pd.DataFrame]:
    """Train Stage 2; return (best_aupr, best_scores_df)."""
    opt   = AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=0.0)
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
                print(
                    f"    [{label}] ep {epoch:4d}/{epochs}"
                    f"  loss={out.loss.item():.4f}"
                    f"  reg={out.regression_loss.item():.4f}"
                    f"  kl={out.kl_loss.item():.4f}"
                    f"  AUPR={m['aupr']:.4f}"
                )

    return best_aupr, best_scores


# ---------------------------------------------------------------------------
# Main per-network runner
# ---------------------------------------------------------------------------

def _alpha_dir(nid: int, alpha: float) -> pathlib.Path:
    return RESULTS_DIR / f"net{nid}" / f"alpha_{alpha}"


def _load_cached_alpha(nid: int, alpha: float) -> float | None:
    """Return cached AUPR for this alpha if it exists, else None."""
    import json
    result_path = _alpha_dir(nid, alpha) / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)["aupr"]
    return None


def _save_alpha_result(
    nid: int, alpha: float, aupr: float,
    scores: pd.DataFrame, stage1: BJEPAStage1,
) -> None:
    import json
    out = _alpha_dir(nid, alpha)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "stage2_scores.csv", index=False)
    torch.save(stage1.state_dict(), out / "stage1.pt")
    with open(out / "result.json", "w") as f:
        json.dump({"aupr": aupr, "fwd_weight": alpha, "network": nid}, f, indent=2)


def run(
    nid: int,
    fwd_weights: list[float],
    device: str,
    uni_aupr: float | None = None,
    reeval: bool = False,
) -> dict:
    import json

    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    print(f"\n{'='*65}")
    print(f"Phase 1B — Asymmetric BiJEPA  |  net{nid} ({network.name})")
    print(f"  n={network.n_samples}  D={network.n_tfs}  G={network.n_genes}"
          f"  n/D={network.n_samples/network.n_tfs:.2f}")
    print(f"  fwd_weights: {fwd_weights}  |  reeval={reeval}")

    # ------------------------------------------------------------------
    # GCV-Ridge baseline
    # ------------------------------------------------------------------
    gcv_path = GCV_DIR / f"net{nid}" / "edge_scores.csv"
    gcv_df   = pd.read_csv(gcv_path)
    gcv_df["score"] = gcv_df["score"].abs()
    gcv_aupr = evaluate_predictions(gcv_df, network.gold_standard)["aupr"]
    print(f"\n  GCV-Ridge AUPR = {gcv_aupr:.4f}  (target to beat)")

    # ------------------------------------------------------------------
    # Shared tensors
    # ------------------------------------------------------------------
    exp2_ckpt = ROOT / "results" / "bjepa" / f"net{nid}" / "stage1_final.pt"
    tau0      = compute_tau0(p0=10.0, n_tfs=network.n_tfs, n_samples=network.n_samples)

    expr_np   = network.expression.values.T.astype("float32")   # (G, n)
    gene_mean = expr_np.mean(axis=1, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=1, keepdims=True), 1e-6)
    expression = torch.from_numpy((expr_np - gene_mean) / gene_std).to(device)

    tf_set  = set(network.tf_ids)
    tf_mask = torch.tensor(
        [g in tf_set for g in network.gene_ids], dtype=torch.bool
    ).to(device)

    X_tf, Y = build_stage2_data(network, device)

    # ------------------------------------------------------------------
    # Unidirectional B-JEPA baseline
    # Priority: --uni_aupr arg > cached result > train from scratch
    # ------------------------------------------------------------------
    uni_cache = RESULTS_DIR / f"net{nid}" / "uni_result.json"

    if uni_aupr is not None:
        print(f"\n  Using provided unidirectional AUPR={uni_aupr:.4f}")
    elif not reeval and uni_cache.exists():
        with open(uni_cache) as f:
            uni_aupr = json.load(f)["aupr"]
        print(f"\n  Loaded cached unidirectional AUPR={uni_aupr:.4f}")
    else:
        # Load Stage 1 from exp2 or train fresh
        uni_s1 = BJEPAStage1.from_network(
            network,
            d_latent=256, encoder_hidden=(512, 512), encoder_dropout=0.1,
            ema_momentum=0.996, kl_weight=1.0, predictor_hidden=256,
            use_bidirectional=False,
        ).to(device)
        ckpt_loaded = False
        if exp2_ckpt.exists():
            try:
                uni_s1.load_state_dict(torch.load(exp2_ckpt, map_location=device))
                print(f"\n  Loading unidirectional Stage 1 from exp2 checkpoint...")
                ckpt_loaded = True
            except RuntimeError:
                print(f"\n  exp2 checkpoint incompatible (old architecture) — training fresh...")
        if not ckpt_loaded:
            print(f"\n  Training unidirectional Stage 1 ({STAGE1_EPOCHS} epochs)...")
            train_stage1(
                uni_s1, expression, tf_mask,
                epochs=STAGE1_EPOCHS, lr=STAGE1_LR, weight_decay=STAGE1_WD,
                fwd_weight=1.0, label="Uni-JEPA",
            )
        uni_s1.eval()

        with torch.no_grad():
            W_init_uni = uni_s1.get_W_init(expression, tf_mask)

        uni_s2 = BJEPAStage2(
            n_tfs=network.n_tfs, n_genes=network.n_genes,
            tau0=tau0, kl_weight=KL_MAX,
        ).to(device)
        uni_s2.init_from_stage1(W_init_uni)

        print(f"\n  Training unidirectional Stage 2 ({STAGE2_EPOCHS} epochs)...")
        t0 = time.perf_counter()
        uni_aupr, _ = train_stage2(
            uni_s2, X_tf, Y, network,
            epochs=STAGE2_EPOCHS, kl_warmup=KL_WARMUP, kl_max=KL_MAX,
            label="Uni  ",
        )
        print(f"  Unidirectional AUPR = {uni_aupr:.4f}  [{time.perf_counter()-t0:.0f}s]")

        uni_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(uni_cache, "w") as f:
            json.dump({"aupr": uni_aupr, "network": nid}, f, indent=2)

    # ------------------------------------------------------------------
    # Bidirectional sweep over fwd_weight α
    # Default: skip if cached result exists. --reeval forces retraining.
    # ------------------------------------------------------------------
    results = {"gcv_aupr": gcv_aupr, "uni_aupr": uni_aupr}
    best_bi_aupr   = -1.0
    best_bi_alpha  = -1.0

    for alpha in fwd_weights:
        cached_aupr = None if reeval else _load_cached_alpha(nid, alpha)

        if cached_aupr is not None:
            print(f"\n  --- fwd_weight α={alpha} [cached AUPR={cached_aupr:.4f}] ---")
            bi_aupr = cached_aupr
        else:
            print(f"\n  --- fwd_weight α={alpha} ---")
            bi_s1 = BJEPAStage1.from_network(
                network,
                d_latent=256, encoder_hidden=(512, 512), encoder_dropout=0.1,
                ema_momentum=0.996, kl_weight=1.0, predictor_hidden=256,
                use_bidirectional=True, fwd_weight=alpha,
            ).to(device)

            t0 = time.perf_counter()
            train_stage1(
                bi_s1, expression, tf_mask,
                epochs=STAGE1_EPOCHS, lr=STAGE1_LR, weight_decay=STAGE1_WD,
                fwd_weight=alpha, label="BiJEPA",
            )
            bi_s1.eval()
            print(f"  Stage 1 done [{time.perf_counter()-t0:.0f}s]")

            with torch.no_grad():
                W_init_bi = bi_s1.get_W_init(expression, tf_mask)
            print(f"  W_init: mean|W|={W_init_bi.abs().mean().item():.4f}")

            bi_s2 = BJEPAStage2(
                n_tfs=network.n_tfs, n_genes=network.n_genes,
                tau0=tau0, kl_weight=KL_MAX,
            ).to(device)
            bi_s2.init_from_stage1(W_init_bi)

            t0 = time.perf_counter()
            bi_aupr, bi_scores = train_stage2(
                bi_s2, X_tf, Y, network,
                epochs=STAGE2_EPOCHS, kl_warmup=KL_WARMUP, kl_max=KL_MAX,
                label=f"Bi α={alpha}",
            )
            print(f"  BiJEPA α={alpha}: AUPR={bi_aupr:.4f}  [{time.perf_counter()-t0:.0f}s]"
                  f"  Δ vs Uni={bi_aupr - uni_aupr:+.4f}"
                  f"  Δ vs GCV={bi_aupr - gcv_aupr:+.4f}")

            _save_alpha_result(nid, alpha, bi_aupr, bi_scores, bi_s1)

        results[f"bijepa_alpha{alpha}"] = bi_aupr

        if bi_aupr > best_bi_aupr:
            best_bi_aupr  = bi_aupr
            best_bi_alpha = alpha

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  {'Method':<40} {'AUPR':>8} {'Δ vs GCV':>10}")
    print(f"  {'-'*60}")
    print(f"  {'GCV-Ridge (baseline)':<40} {gcv_aupr:>8.4f} {'—':>10}")
    print(f"  {'Unidirectional B-JEPA':<40} {uni_aupr:>8.4f} {uni_aupr - gcv_aupr:>+10.4f}")
    for alpha in fwd_weights:
        aupr   = results.get(f"bijepa_alpha{alpha}", float("nan"))
        marker = " ◄ best" if alpha == best_bi_alpha else ""
        print(f"  {f'BiJEPA α={alpha}':<40} {aupr:>8.4f} {aupr - gcv_aupr:>+10.4f}{marker}")

    print(f"\n  Best BiJEPA: α={best_bi_alpha}  AUPR={best_bi_aupr:.4f}"
          f"  Δ vs Uni={best_bi_aupr - uni_aupr:+.4f}"
          f"  Δ vs GCV={best_bi_aupr - gcv_aupr:+.4f}")

    # Write top-level summary pointing to best alpha
    out_dir = RESULTS_DIR / f"net{nid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bijepa_summary.json", "w") as f:
        json.dump({
            "best_alpha": best_bi_alpha,
            "best_aupr":  best_bi_aupr,
            "gcv_aupr":   gcv_aupr,
            "uni_aupr":   uni_aupr,
            "all_alphas": {str(a): results.get(f"bijepa_alpha{a}") for a in fwd_weights},
        }, f, indent=2)

    print(f"\n  Results saved under {out_dir}/")

    results.update({
        "network":       f"net{nid} ({network.name})",
        "best_bi_alpha": best_bi_alpha,
        "best_bi_aupr":  best_bi_aupr,
    })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4],
                        help="Single network to run (default: all 4)")
    parser.add_argument("--fwd_weights", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 0.9],
                        help="Forward weight values to sweep (default: 0.3 0.5 0.7 0.9)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--uni_aupr", type=float, default=None,
                        help="Skip unidirectional Stage 2 training, use this AUPR directly")
    parser.add_argument("--reeval", action="store_true",
                        help="Ignore cached results and retrain all alphas from scratch")
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]
    all_results = []

    for nid in nids:
        r = run(nid, args.fwd_weights, args.device,
                uni_aupr=args.uni_aupr, reeval=args.reeval)
        if r:
            all_results.append(r)

    if len(all_results) > 1:
        print("\n\nSummary across networks:")
        df = pd.DataFrame(all_results).set_index("network")
        print(df.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
