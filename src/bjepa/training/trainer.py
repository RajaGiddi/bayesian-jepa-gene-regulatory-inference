"""B-JEPA two-stage trainer.

Stage 1: Train context + target encoders via JEPA latent prediction loss.
Stage 2: Freeze encoders, infer sparse regulatory weights W via horseshoe VI.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from bjepa.data.dream5 import DREAM5Network
from bjepa.eval.metrics import evaluate_predictions
from bjepa.models.bjepa import BJEPAStage1, BJEPAStage2

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    # Stage 1
    stage1_epochs:         int   = 150
    stage1_lr:             float = 3e-4
    stage1_weight_decay:   float = 1e-4

    # Stage 2
    stage2_epochs:         int   = 200
    stage2_lr:             float = 1e-3    # higher LR — only W is trained
    stage2_weight_decay:   float = 0.0
    kl_warmup_epochs:      int   = 50      # β: 0→kl_weight_max over this many epochs
    kl_weight_max:         float = 0.1     # max KL weight after warmup (1.0 is too tight)
    kl_mc_samples:         int   = 4

    # Shared
    eval_every:            int   = 10
    results_dir:           str   = "results/bjepa"
    device:                str   = "cpu"


@dataclass
class TrainHistory:
    # Stage 1
    s1_loss:     list[float] = field(default_factory=list)
    # Stage 2
    s2_reg_loss: list[float] = field(default_factory=list)
    s2_kl_loss:  list[float] = field(default_factory=list)
    s2_kl_weight: list[float] = field(default_factory=list)
    # Eval (stage 2)
    auroc:       list[float] = field(default_factory=list)
    aupr:        list[float] = field(default_factory=list)
    eval_epochs: list[int]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BJEPATrainer:
    def __init__(
        self,
        stage1: BJEPAStage1,
        stage2: BJEPAStage2,
        network: DREAM5Network,
        cfg: TrainerConfig,
    ) -> None:
        self.stage1  = stage1.to(cfg.device)
        self.stage2  = stage2.to(cfg.device)
        self.network = network
        self.cfg     = cfg
        self.history = TrainHistory()

        self.results_dir = pathlib.Path(cfg.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Fixed input tensors — standardise per gene (zero-mean, unit-std across samples)
        # so the encoder learns variation patterns, not mean expression level.
        expr_np = network.expression.values.T.astype(np.float32)   # (n_genes, n_samples)
        gene_mean = expr_np.mean(axis=1, keepdims=True)
        gene_std  = np.maximum(expr_np.std(axis=1, keepdims=True), 1e-6)
        self.expression = torch.from_numpy((expr_np - gene_mean) / gene_std).to(cfg.device)

        tf_set = set(network.tf_ids)
        self.tf_mask = torch.tensor(
            [g in tf_set for g in network.gene_ids], dtype=torch.bool
        ).to(cfg.device)

        self._best_aupr  = -1.0
        self._best_epoch = -1

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def train_stage1(self) -> None:
        cfg    = self.cfg
        model  = self.stage1

        opt = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.stage1_lr, weight_decay=cfg.stage1_weight_decay,
        )
        sched = CosineAnnealingLR(opt, T_max=cfg.stage1_epochs, eta_min=cfg.stage1_lr * 0.1)

        print(f"\n--- Stage 1: JEPA representation learning ({cfg.stage1_epochs} epochs) ---")
        t0 = time.perf_counter()

        for epoch in range(1, cfg.stage1_epochs + 1):
            model.train()
            opt.zero_grad()
            out = model(self.expression, self.tf_mask)
            out.loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            opt.step()
            sched.step()
            model.update_ema()

            self.history.s1_loss.append(out.loss.item())

            if epoch % cfg.eval_every == 0 or epoch == cfg.stage1_epochs:
                elapsed = time.perf_counter() - t0
                print(f"  S1 Epoch {epoch:4d}/{cfg.stage1_epochs}  jepa={out.loss.item():.4f}  [{elapsed:.0f}s]")

        # Save stage 1 checkpoint
        torch.save(model.state_dict(), self.results_dir / "stage1_final.pt")
        print(f"Stage 1 complete. Checkpoint saved.")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def train_stage2(self) -> None:
        cfg   = self.cfg
        s1    = self.stage1
        model = self.stage2

        # Compute fixed embeddings once (encoders frozen)
        print(f"\n--- Stage 2: Horseshoe regression in expression space ({cfg.stage2_epochs} epochs) ---")

        # Warm-start mu_W from Stage 1's W_init
        W_init = s1.get_W_init()
        model.init_from_stage1(W_init)
        print(f"  Warm-started mu_W from Stage 1 W_init (norm={W_init.norm():.3f})")

        # Build fixed expression matrices for Stage 2 (standardised per gene)
        # X_tf: (n_samples, n_tfs), Y: (n_samples, n_genes)
        tf_indices = [i for i, g in enumerate(self.network.gene_ids)
                      if g in set(self.network.tf_ids)]
        expr_np   = self.network.expression.values.astype("float32")  # (n_samples, n_genes)
        gene_mean = expr_np.mean(axis=0, keepdims=True)                # (1, n_genes)
        gene_std  = np.maximum(expr_np.std(axis=0, keepdims=True), 1e-6)
        expr_std  = (expr_np - gene_mean) / gene_std                   # (n_samples, n_genes), unit variance
        X_tf = torch.from_numpy(expr_std[:, tf_indices]).to(cfg.device)
        Y    = torch.from_numpy(expr_std).to(cfg.device)

        print(f"  X_tf: {tuple(X_tf.shape)}  Y: {tuple(Y.shape)}")
        print(f"  Regression: {self.network.n_tfs} TFs × {self.network.n_samples} samples → {self.network.n_genes} genes")

        opt = AdamW(
            model.parameters(),
            lr=cfg.stage2_lr, weight_decay=cfg.stage2_weight_decay,
        )
        sched = CosineAnnealingLR(opt, T_max=cfg.stage2_epochs, eta_min=cfg.stage2_lr * 0.05)

        t0 = time.perf_counter()

        for epoch in range(1, cfg.stage2_epochs + 1):
            model.train()
            kl_weight = min(cfg.kl_weight_max, epoch / max(1, cfg.kl_warmup_epochs) * cfg.kl_weight_max)
            model.kl_weight = kl_weight

            opt.zero_grad()
            out = model(X_tf, Y, kl_mc_samples=cfg.kl_mc_samples)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            self.history.s2_reg_loss.append(out.regression_loss.item())
            self.history.s2_kl_loss.append(out.kl_loss.item())
            self.history.s2_kl_weight.append(kl_weight)

            if epoch % cfg.eval_every == 0 or epoch == cfg.stage2_epochs:
                metrics = self._evaluate()
                elapsed = time.perf_counter() - t0
                print(
                    f"  S2 Epoch {epoch:4d}/{cfg.stage2_epochs}"
                    f"  reg={out.regression_loss.item():.4f}"
                    f"  kl={out.kl_loss.item():.4f}"
                    f"  β={kl_weight:.2f}"
                    f"  AUROC={metrics['auroc']:.4f}"
                    f"  AUPR={metrics['aupr']:.4f}"
                    f"  [{elapsed:.0f}s]"
                )
                self.history.auroc.append(metrics["auroc"])
                self.history.aupr.append(metrics["aupr"])
                self.history.eval_epochs.append(epoch)

                if metrics["aupr"] > self._best_aupr:
                    self._best_aupr  = metrics["aupr"]
                    self._best_epoch = epoch
                    torch.save(model.state_dict(), self.results_dir / "stage2_best.pt")

        torch.save(model.state_dict(), self.results_dir / "stage2_final.pt")
        print(f"\nBest AUPR={self._best_aupr:.4f} at stage-2 epoch {self._best_epoch}")

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------

    def train(self) -> TrainHistory:
        net = self.network
        print(f"\nB-JEPA two-stage training: net{net.network_id} ({net.name})")
        print(f"  {net.n_samples} samples | {net.n_genes} genes | {net.n_tfs} TFs | device: {self.cfg.device}")

        self.train_stage1()
        self.train_stage2()

        self._save_history()
        return self.history

    # ------------------------------------------------------------------
    # Evaluation + I/O
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> dict:
        if self.network.gold_standard is None:
            return {"auroc": float("nan"), "aupr": float("nan")}
        self.stage2.eval()
        scores_df = self.stage2.edge_scores(self.network.gene_ids, self.network.tf_ids)
        scores_df = scores_df.copy()
        scores_df["score"] = scores_df["score"].abs()
        return evaluate_predictions(scores_df, self.network.gold_standard)

    def _save_history(self) -> None:
        with open(self.results_dir / "history.json", "w") as f:
            json.dump(asdict(self.history), f, indent=2)
