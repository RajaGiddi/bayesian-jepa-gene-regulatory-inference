"""Phase 1C: Stochastic Masking JEPA (SM-JEPA).

Instead of a fixed TF-as-context / non-TF-as-target split, each training
step stochastically reassigns context and target roles with TF-biased
probability:

    p(context | TF gene)     = tf_bias
    p(context | non-TF gene) = 1 - tf_bias

tf_bias=1.0  →  TFs always context  →  standard B-JEPA (control)
tf_bias=0.5  →  fully symmetric, no causal prior

Why this improves on BiJEPA
---------------------------
BiJEPA adds a separate backward predictor with its own parameters.  At n=160
the backward regression (2711 non-TF genes → 99 TF targets) is too noisy to
learn from.  SM-JEPA achieves bidirectional consistency without a second model:
the shared encoder must produce representations that are predictable from ANY
sufficient subset of genes, not just TFs.  The tf_bias controls how much the
causal direction is favoured.

At eval time and for W_init extraction, the standard TF-as-context assignment
is always used — stochasticity is training-only.

Sweep
-----
tf_bias ∈ {0.5, 0.6, 0.7, 0.8, 1.0}

For each tf_bias:
  1. Train SM-JEPA Stage 1 from scratch (150 epochs).
  2. Extract W_init via OLS (same as all other experiments).
  3. Train Linear Stage 2 (300 epochs, kl_max=0.05).
  4. Report AUPR vs GCV-Ridge, unidirectional B-JEPA, and best BiJEPA.

Usage:
    python experiments/exp_smjepa/run_smjepa.py --network 2 --device mps
    python experiments/exp_smjepa/run_smjepa.py --network 2 --tf_biases 0.6 0.7 0.8
    python experiments/exp_smjepa/run_smjepa.py --network 2 --reeval --device mps
"""
from __future__ import annotations

import argparse
import json
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
BJEPA_DIR   = ROOT / "results" / "bjepa"
BIJEPA_DIR  = ROOT / "results" / "bijepa"
RESULTS_DIR = ROOT / "results" / "smjepa"

STAGE1_EPOCHS = 150
STAGE1_LR     = 3e-4
STAGE1_WD     = 1e-4
STAGE2_EPOCHS = 300
STAGE2_LR     = 1e-3
KL_WARMUP     = 50
KL_MAX        = 0.05


# ---------------------------------------------------------------------------
# Caching helpers (mirrors run_bijepa.py pattern)
# ---------------------------------------------------------------------------

def _bias_dir(nid: int, tf_bias: float) -> pathlib.Path:
    return RESULTS_DIR / f"net{nid}" / f"bias_{tf_bias}"


def _load_cached(nid: int, tf_bias: float) -> float | None:
    p = _bias_dir(nid, tf_bias) / "result.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)["aupr"]
    return None


def _save_result(
    nid: int, tf_bias: float, aupr: float,
    scores: pd.DataFrame, stage1: BJEPAStage1,
) -> None:
    out = _bias_dir(nid, tf_bias)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "stage2_scores.csv", index=False)
    torch.save(stage1.state_dict(), out / "stage1.pt")
    with open(out / "result.json", "w") as f:
        json.dump({"aupr": aupr, "tf_bias": tf_bias, "network": nid}, f, indent=2)


# ---------------------------------------------------------------------------
# Training helpers (same structure as run_bijepa.py)
# ---------------------------------------------------------------------------

def build_stage1_tensors(network, device: str):
    expr_np   = network.expression.values.T.astype("float32")
    gene_mean = expr_np.mean(axis=1, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=1, keepdims=True), 1e-6)
    expression = torch.from_numpy((expr_np - gene_mean) / gene_std).to(device)
    tf_set  = set(network.tf_ids)
    tf_mask = torch.tensor(
        [g in tf_set for g in network.gene_ids], dtype=torch.bool
    ).to(device)
    return expression, tf_mask


def build_stage2_data(network, device: str):
    expr_np   = network.expression.values.astype("float32")
    gene_mean = expr_np.mean(axis=0, keepdims=True)
    gene_std  = np.maximum(expr_np.std(axis=0, keepdims=True), 1e-6)
    expr_std  = (expr_np - gene_mean) / gene_std
    tf_indices = [i for i, g in enumerate(network.gene_ids)
                  if g in set(network.tf_ids)]
    X_tf = torch.from_numpy(expr_std[:, tf_indices]).to(device)
    Y    = torch.from_numpy(expr_std).to(device)
    return X_tf, Y


def train_stage1(model, expression, tf_mask, tf_bias, epochs, lr, wd):
    opt   = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
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
            print(
                f"    [SM-JEPA bias={tf_bias}] ep {epoch:4d}/{epochs}"
                f"  loss={out.loss.item():.4f}"
                f"  nll={out.nll.item():.4f}"
                f"  kl={out.kl.item():.4f}"
            )


def train_stage2(model, X_tf, Y, network, label):
    opt   = AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS, eta_min=5e-5)
    best_aupr, best_scores = -1.0, None
    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        kl_w = min(KL_MAX, epoch / max(1, KL_WARMUP) * KL_MAX)
        model.kl_weight = kl_w
        opt.zero_grad()
        out = model(X_tf, Y, kl_mc_samples=4)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        if epoch % 10 == 0 or epoch == STAGE2_EPOCHS:
            model.eval()
            with torch.no_grad():
                scores = model.edge_scores(network.gene_ids, network.tf_ids)
            scores["score"] = scores["score"].abs()
            m = evaluate_predictions(scores, network.gold_standard)
            if m["aupr"] > best_aupr:
                best_aupr, best_scores = m["aupr"], scores.copy()
            if epoch % 50 == 0 or epoch == STAGE2_EPOCHS:
                print(
                    f"    [{label}] ep {epoch:4d}/{STAGE2_EPOCHS}"
                    f"  loss={out.loss.item():.4f}"
                    f"  reg={out.regression_loss.item():.4f}"
                    f"  AUPR={m['aupr']:.4f}"
                )
    return best_aupr, best_scores


# ---------------------------------------------------------------------------
# Per-network runner
# ---------------------------------------------------------------------------

def run(
    nid: int,
    tf_biases: list[float],
    device: str,
    uni_aupr: float | None = None,
    reeval: bool = False,
) -> dict:
    network = load_network(DATA_DIR, nid)
    if network.gold_standard is None:
        print(f"net{nid}: no gold standard, skipping.")
        return {}

    print(f"\n{'='*65}")
    print(f"Phase 1C — SM-JEPA  |  net{nid} ({network.name})")
    print(f"  n={network.n_samples}  D={network.n_tfs}  G={network.n_genes}"
          f"  n/D={network.n_samples/network.n_tfs:.2f}")
    print(f"  tf_biases: {tf_biases}  |  reeval={reeval}")

    # GCV baseline
    gcv_df   = pd.read_csv(GCV_DIR / f"net{nid}" / "edge_scores.csv")
    gcv_df["score"] = gcv_df["score"].abs()
    gcv_aupr = evaluate_predictions(gcv_df, network.gold_standard)["aupr"]
    print(f"\n  GCV-Ridge AUPR = {gcv_aupr:.4f}")

    # Load BiJEPA best result for comparison (if available)
    bijepa_summary = BIJEPA_DIR / f"net{nid}" / "bijepa_summary.json"
    bijepa_best_aupr = None
    bijepa_best_alpha = None
    if bijepa_summary.exists():
        with open(bijepa_summary) as f:
            s = json.load(f)
        bijepa_best_aupr  = s.get("best_aupr")
        bijepa_best_alpha = s.get("best_alpha")
        print(f"  BiJEPA best AUPR = {bijepa_best_aupr:.4f}  (α={bijepa_best_alpha})")

    tau0       = compute_tau0(p0=10.0, n_tfs=network.n_tfs, n_samples=network.n_samples)
    expression, tf_mask = build_stage1_tensors(network, device)
    X_tf, Y    = build_stage2_data(network, device)

    # ------------------------------------------------------------------
    # Unidirectional baseline
    # Priority: --uni_aupr > cached > train from scratch
    # ------------------------------------------------------------------
    uni_cache = RESULTS_DIR / f"net{nid}" / "uni_result.json"

    if uni_aupr is not None:
        print(f"\n  Using provided unidirectional AUPR={uni_aupr:.4f}")
    elif not reeval and uni_cache.exists():
        with open(uni_cache) as f:
            uni_aupr = json.load(f)["aupr"]
        print(f"\n  Loaded cached unidirectional AUPR={uni_aupr:.4f}")
    else:
        exp2_ckpt = BJEPA_DIR / f"net{nid}" / "stage1_final.pt"
        uni_s1 = BJEPAStage1.from_network(
            network, d_latent=256, encoder_hidden=(512, 512),
            encoder_dropout=0.1, ema_momentum=0.996, kl_weight=1.0,
        ).to(device)
        ckpt_loaded = False
        if exp2_ckpt.exists():
            try:
                uni_s1.load_state_dict(torch.load(exp2_ckpt, map_location=device))
                print(f"\n  Loading unidirectional Stage 1 from exp2...")
                ckpt_loaded = True
            except RuntimeError:
                print(f"\n  exp2 checkpoint incompatible (old architecture) — training fresh...")
        if not ckpt_loaded:
            print(f"\n  Training unidirectional Stage 1 ({STAGE1_EPOCHS} epochs)...")
            train_stage1(uni_s1, expression, tf_mask, tf_bias=1.0,
                         epochs=STAGE1_EPOCHS, lr=STAGE1_LR, wd=STAGE1_WD)
        uni_s1.eval()

        with torch.no_grad():
            W_init = uni_s1.get_W_init(expression, tf_mask)
        uni_s2 = BJEPAStage2(
            n_tfs=network.n_tfs, n_genes=network.n_genes,
            tau0=tau0, kl_weight=KL_MAX,
        ).to(device)
        uni_s2.init_from_stage1(W_init)

        print(f"  Training unidirectional Stage 2 ({STAGE2_EPOCHS} epochs)...")
        t0 = time.perf_counter()
        uni_aupr, _ = train_stage2(uni_s2, X_tf, Y, network, label="Uni  ")
        print(f"  Unidirectional AUPR = {uni_aupr:.4f}  [{time.perf_counter()-t0:.0f}s]")

        uni_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(uni_cache, "w") as f:
            json.dump({"aupr": uni_aupr, "network": nid}, f, indent=2)

    # ------------------------------------------------------------------
    # SM-JEPA sweep
    # ------------------------------------------------------------------
    results = {"gcv_aupr": gcv_aupr, "uni_aupr": uni_aupr}
    best_aupr, best_bias = -1.0, -1.0

    for tf_bias in tf_biases:
        cached = None if reeval else _load_cached(nid, tf_bias)

        if cached is not None:
            print(f"\n  --- tf_bias={tf_bias} [cached AUPR={cached:.4f}] ---")
            sm_aupr = cached
        else:
            print(f"\n  --- tf_bias={tf_bias} ---")
            sm_s1 = BJEPAStage1.from_network(
                network, d_latent=256, encoder_hidden=(512, 512),
                encoder_dropout=0.1, ema_momentum=0.996, kl_weight=1.0,
                use_stochastic_mask=True, tf_bias=tf_bias,
            ).to(device)

            t0 = time.perf_counter()
            train_stage1(sm_s1, expression, tf_mask, tf_bias=tf_bias,
                         epochs=STAGE1_EPOCHS, lr=STAGE1_LR, wd=STAGE1_WD)
            sm_s1.eval()
            print(f"  Stage 1 done [{time.perf_counter()-t0:.0f}s]")

            with torch.no_grad():
                W_init = sm_s1.get_W_init(expression, tf_mask)
            print(f"  W_init: mean|W|={W_init.abs().mean().item():.4f}")

            sm_s2 = BJEPAStage2(
                n_tfs=network.n_tfs, n_genes=network.n_genes,
                tau0=tau0, kl_weight=KL_MAX,
            ).to(device)
            sm_s2.init_from_stage1(W_init)

            t0 = time.perf_counter()
            sm_aupr, sm_scores = train_stage2(
                sm_s2, X_tf, Y, network, label=f"SM b={tf_bias}"
            )
            print(f"  SM-JEPA bias={tf_bias}: AUPR={sm_aupr:.4f}  [{time.perf_counter()-t0:.0f}s]"
                  f"  Δ vs Uni={sm_aupr - uni_aupr:+.4f}"
                  f"  Δ vs GCV={sm_aupr - gcv_aupr:+.4f}")

            _save_result(nid, tf_bias, sm_aupr, sm_scores, sm_s1)

        results[f"sm_bias{tf_bias}"] = sm_aupr
        if sm_aupr > best_aupr:
            best_aupr, best_bias = sm_aupr, tf_bias

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  {'Method':<42} {'AUPR':>8} {'Δ vs GCV':>10}")
    print(f"  {'-'*62}")
    print(f"  {'GCV-Ridge':<42} {gcv_aupr:>8.4f} {'—':>10}")
    print(f"  {'Unidirectional B-JEPA':<42} {uni_aupr:>8.4f} {uni_aupr - gcv_aupr:>+10.4f}")
    if bijepa_best_aupr is not None:
        print(f"  {f'BiJEPA best (α={bijepa_best_alpha})':<42} "
              f"{bijepa_best_aupr:>8.4f} {bijepa_best_aupr - gcv_aupr:>+10.4f}")
    for tf_bias in tf_biases:
        aupr   = results.get(f"sm_bias{tf_bias}", float("nan"))
        label  = f"SM-JEPA bias={tf_bias}" + (" (=B-JEPA)" if tf_bias == 1.0 else "")
        marker = " ◄ best" if tf_bias == best_bias else ""
        print(f"  {label:<42} {aupr:>8.4f} {aupr - gcv_aupr:>+10.4f}{marker}")

    print(f"\n  Best SM-JEPA: bias={best_bias}  AUPR={best_aupr:.4f}"
          f"  Δ vs Uni={best_aupr - uni_aupr:+.4f}"
          f"  Δ vs GCV={best_aupr - gcv_aupr:+.4f}")

    out_dir = RESULTS_DIR / f"net{nid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "smjepa_summary.json", "w") as f:
        json.dump({
            "best_bias":  best_bias,
            "best_aupr":  best_aupr,
            "gcv_aupr":   gcv_aupr,
            "uni_aupr":   uni_aupr,
            "all_biases": {str(b): results.get(f"sm_bias{b}") for b in tf_biases},
        }, f, indent=2)

    print(f"\n  Results saved under {out_dir}/")
    results.update({
        "network":     f"net{nid} ({network.name})",
        "best_bias":   best_bias,
        "best_aupr":   best_aupr,
    })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--tf_biases", type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8, 1.0],
                        help="TF context probability values to sweep (default: 0.5 0.6 0.7 0.8 1.0)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--uni_aupr", type=float, default=None,
                        help="Skip unidirectional Stage 2, use this AUPR directly")
    parser.add_argument("--reeval", action="store_true",
                        help="Ignore cached results and retrain all tf_bias values")
    args = parser.parse_args()

    nids = [args.network] if args.network else [1, 2, 3, 4]
    all_results = []
    for nid in nids:
        r = run(nid, args.tf_biases, args.device,
                uni_aupr=args.uni_aupr, reeval=args.reeval)
        if r:
            all_results.append(r)

    if len(all_results) > 1:
        print("\n\nSummary across networks:")
        df = pd.DataFrame(all_results).set_index("network")
        print(df.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
