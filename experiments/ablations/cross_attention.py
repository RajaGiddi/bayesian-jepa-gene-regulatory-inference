"""Ablation 3: Cross-attention predictor in Stage 1.

Replaces mean-pool context aggregation with multi-head cross-attention so
each gene attends selectively to individual TF latents.  Tests three score
sources and their ensemble with GCV-Ridge:

  (A) Attention weights directly as edge scores   — no Stage 2 needed
  (B) Stage 2 horseshoe VI with cross-attn W_init  — same as exp2 but better warm start
  (C) Ensemble: GCV-Ridge + (A) attention scores
  (D) Ensemble: GCV-Ridge + (B) Stage 2 scores

Hypothesis: cross-attention forces per-TF selectivity, so attention weights
encode regulatory signal directly.  Expected to fix net3 where mean-pool
loses selectivity across 334 TFs.

Usage (from repo root):
    python experiments/ablations/cross_attention.py --network 2
    python experiments/ablations/cross_attention.py --network 3
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
RESULTS_DIR = ROOT / "results" / "ablations" / "cross_attention"
GCV_DIR   = ROOT / "results" / "analytical_hs" / "ols"


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["score"].rank(method="average") / len(df)
    return df


def ensemble_scores(a: pd.DataFrame, b: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """alpha = weight on `a`, (1-alpha) on `b`. Both must be rank-normalised."""
    m = a.merge(b, on=["tf", "target"], suffixes=("_a", "_b"))
    m["score"] = alpha * m["score_a"] + (1 - alpha) * m["score_b"]
    return m[["tf", "target", "score"]]


def build_expression(network, device: str):
    expr_np   = network.expression.values.T.astype("float32")  # (n_genes, n_samples)
    gene_mean = expr_np.mean(axis=1, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=1, keepdims=True), 1e-6)
    expr_t    = torch.from_numpy((expr_np - gene_mean) / gene_std).to(device)
    tf_set    = set(network.tf_ids)
    tf_mask   = torch.tensor([g in tf_set for g in network.gene_ids],
                              dtype=torch.bool).to(device)
    return expr_t, tf_mask


def build_stage2_data(network, device: str):
    expr_np   = network.expression.values.astype("float32")   # (n_samples, n_genes)
    gene_mean = expr_np.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=0, keepdims=True), 1e-6)
    expr_std  = (expr_np - gene_mean) / gene_std
    tf_indices = [i for i, g in enumerate(network.gene_ids) if g in set(network.tf_ids)]
    X_tf = torch.from_numpy(expr_std[:, tf_indices]).to(device)
    Y    = torch.from_numpy(expr_std).to(device)
    return X_tf, Y


def train_stage1(model: BJEPAStage1, expression, tf_mask, epochs: int,
                 lr: float, weight_decay: float, eval_every: int) -> None:
    opt   = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    print(f"\n  Stage 1 (cross-attention, {epochs} epochs):")
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(expression, tf_mask)
        out.loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        opt.step()
        sched.step()
        model.update_ema()

        if epoch % eval_every == 0 or epoch == epochs:
            print(f"    ep {epoch:4d}/{epochs}  elbo={out.loss.item():.4f}"
                  f"  nll={out.nll.item():.4f}  kl={out.kl.item():.4f}"
                  f"  [{time.perf_counter()-t0:.0f}s]")


def train_stage2(stage2: BJEPAStage2, X_tf, Y, network,
                 epochs: int, kl_warmup: int, kl_max: float) -> float:
    opt   = AdamW(stage2.parameters(), lr=1e-3, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)
    best_aupr = -1.0

    for epoch in range(1, epochs + 1):
        stage2.train()
        kl_w = min(kl_max, epoch / max(1, kl_warmup) * kl_max)
        stage2.kl_weight = kl_w
        opt.zero_grad()
        out = stage2(X_tf, Y, kl_mc_samples=4)
        out.loss.backward()
        nn.utils.clip_grad_norm_(stage2.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if epoch % 10 == 0 or epoch == epochs:
            stage2.eval()
            with torch.no_grad():
                scores = stage2.edge_scores(network.gene_ids, network.tf_ids)
            scores["score"] = scores["score"].abs()
            m = evaluate_predictions(scores, network.gold_standard)
            if m["aupr"] > best_aupr:
                best_aupr = m["aupr"]
                best_scores = scores.copy()

    return best_aupr, best_scores


def run(nid: int, s1_epochs: int, s2_epochs: int, device: str,
        num_heads: int, alpha_gcv: float) -> None:
    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Ablation 3 — Cross-attention  |  net{nid} ({network.name})")
    print(f"  {network.n_samples} samples | {network.n_tfs} TFs | "
          f"{network.n_genes} genes | heads={num_heads}")

    # GCV-Ridge baseline
    gcv_path = GCV_DIR / f"net{nid}" / "edge_scores.csv"
    gcv_df   = pd.read_csv(gcv_path)
    gcv_df["score"] = gcv_df["score"].abs()
    gcv_aupr = evaluate_predictions(gcv_df, network.gold_standard)["aupr"]
    print(f"\n  GCV-Ridge AUPR = {gcv_aupr:.4f}")

    expr_t, tf_mask = build_expression(network, device)
    X_tf, Y         = build_stage2_data(network, device)

    # ------------------------------------------------------------------ #
    # Stage 1: train with cross-attention
    # ------------------------------------------------------------------ #
    stage1 = BJEPAStage1.from_network(
        network,
        d_latent=256, encoder_hidden=(512, 512), encoder_dropout=0.1,
        ema_momentum=0.996, kl_weight=1.0, predictor_hidden=256,
        use_cross_attention=True, num_attn_heads=num_heads,
    ).to(device)

    train_stage1(stage1, expr_t, tf_mask,
                 epochs=s1_epochs, lr=3e-4, weight_decay=1e-4, eval_every=30)

    # ------------------------------------------------------------------ #
    # (A) Attention weights as direct edge scores
    # ------------------------------------------------------------------ #
    stage1.eval()
    attn_scores = stage1.get_attention_scores(
        expr_t, tf_mask, network.gene_ids, network.tf_ids)
    m_attn = evaluate_predictions(attn_scores, network.gold_standard)
    print(f"\n  (A) Attention scores (direct):  AUPR={m_attn['aupr']:.4f}"
          f"  AUROC={m_attn['auroc']:.4f}")

    # ------------------------------------------------------------------ #
    # (B) Stage 2 with cross-attn W_init
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        W_init = stage1.get_W_init(expr_t, tf_mask)

    tau0   = compute_tau0(p0=10.0, n_tfs=network.n_tfs, n_samples=network.n_samples)
    stage2 = BJEPAStage2(n_tfs=network.n_tfs, n_genes=network.n_genes,
                         tau0=tau0, kl_weight=0.1).to(device)
    stage2.init_from_stage1(W_init)

    print(f"\n  Stage 2 ({s2_epochs} epochs):")
    t0 = time.perf_counter()
    best_s2_aupr, s2_scores = train_stage2(
        stage2, X_tf, Y, network,
        epochs=s2_epochs, kl_warmup=50, kl_max=0.1)
    print(f"  (B) Stage 2 (cross-attn W_init): AUPR={best_s2_aupr:.4f}"
          f"  [{time.perf_counter()-t0:.0f}s]")

    # ------------------------------------------------------------------ #
    # (C) Ensemble: GCV + attention scores
    # ------------------------------------------------------------------ #
    gcv_rn  = rank_normalise(gcv_df)
    attn_rn = rank_normalise(attn_scores)
    s2_rn   = rank_normalise(s2_scores)

    # Sweep alpha to find best GCV + attention ensemble
    best_c = {"aupr": -1.0, "alpha": -1.0}
    best_d = {"aupr": -1.0, "alpha": -1.0}
    for alpha in np.linspace(0.0, 1.0, 21):
        mc = evaluate_predictions(ensemble_scores(gcv_rn, attn_rn, alpha),
                                  network.gold_standard)
        md = evaluate_predictions(ensemble_scores(gcv_rn, s2_rn, alpha),
                                  network.gold_standard)
        if mc["aupr"] > best_c["aupr"]:
            best_c = {"aupr": mc["aupr"], "alpha": round(float(alpha), 2)}
        if md["aupr"] > best_d["aupr"]:
            best_d = {"aupr": md["aupr"], "alpha": round(float(alpha), 2)}

    print(f"\n  (C) Ensemble GCV+Attention:  AUPR={best_c['aupr']:.4f}"
          f"  (α_gcv={best_c['alpha']})")
    print(f"  (D) Ensemble GCV+Stage2:     AUPR={best_d['aupr']:.4f}"
          f"  (α_gcv={best_d['alpha']})")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n  {'Method':<30} {'AUPR':>8} {'vs GCV':>8}")
    print(f"  {'-'*48}")
    for label, aupr in [
        ("GCV-Ridge (baseline)",    gcv_aupr),
        ("(A) Attention scores",     m_attn["aupr"]),
        ("(B) Stage 2 cross-attn",   best_s2_aupr),
        ("(C) GCV + Attention",       best_c["aupr"]),
        ("(D) GCV + Stage 2",         best_d["aupr"]),
    ]:
        delta = aupr - gcv_aupr
        print(f"  {label:<30} {aupr:>8.4f} {delta:>+8.4f}")

    # Save checkpoint
    out_dir = RESULTS_DIR / f"net{nid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stage1.state_dict(), out_dir / "stage1_xattn.pt")
    attn_scores.to_csv(out_dir / "attn_scores.csv", index=False)
    s2_scores.to_csv(out_dir / "stage2_scores.csv", index=False)
    print(f"\n  Saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",   type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--s1_epochs", type=int, default=150)
    parser.add_argument("--s2_epochs", type=int, default=200)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--alpha_gcv", type=float, default=0.65)
    parser.add_argument("--device",    type=str, default="mps")
    args = parser.parse_args()
    run(args.network, args.s1_epochs, args.s2_epochs,
        args.device, args.num_heads, args.alpha_gcv)


if __name__ == "__main__":
    main()
