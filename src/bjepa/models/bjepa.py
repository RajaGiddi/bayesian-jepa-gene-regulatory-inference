"""B-JEPA: two-stage architecture.

Stage 1 — JEPA encoder training with linear W predictor
---------------------------------------------------------
Train context + target encoders so that linear combinations of TF latents
reconstruct target gene latents. W_init carries a rough regulatory signal.

    pred_s_target = (s_context.T @ W_init).T
    L_stage1 = MSE(pred_s_target, stop_grad(s_target))

Stage 2 — Horseshoe sparse regression in expression space
----------------------------------------------------------
Freeze encoders. Use W_init as a warm start for mu_W.

Regress raw gene expression on raw TF expression with horseshoe prior:

    pred = X_tf @ W          (n_samples, n_genes)
    L = MSE(pred, Y) + β · KL(q(W) || p_horseshoe)

Why expression space (not latent space)
----------------------------------------
Regressing s_target[g] ∈ R^d on s_context ∈ R^{n_tfs × d} is underdetermined
when d > n_tfs: infinitely many W satisfy the constraint and the horseshoe
finds the minimum-norm solution, which is not the sparse biological one.

With raw expression: n_samples (805) >> n_tfs (195) so the problem is
well-determined and the horseshoe uniquely identifies which TFs co-vary with
each gene across experimental conditions. The JEPA's value is the warm-started
W_init, which gives better initialisation than BiGSM's cold random start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ContextEncoder, TargetEncoder, build_encoders
from .horseshoe import HorseshoeRegressor, compute_tau0


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class Stage1Output:
    pred_s_target: torch.Tensor  # (n_genes, d_latent)
    s_target:      torch.Tensor  # (n_genes, d_latent)
    s_context:     torch.Tensor  # (n_tfs,  d_latent)
    loss:          torch.Tensor  # scalar MSE


@dataclass
class Stage2Output:
    regression_loss: torch.Tensor  # MSE in expression space
    kl_loss:         torch.Tensor
    loss:            torch.Tensor


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class BJEPAStage1(nn.Module):
    """JEPA encoder training with linear W predictor.

    pred_s_target[g] = sum_i W_init[i,g] * s_context[i]
                     = (s_context.T @ W_init).T

    W_init is regularised by weight_decay (Gaussian L2 prior), identical in
    form to the horseshoe's slab component before Stage 2 refines it.
    """

    def __init__(
        self,
        n_samples: int,
        n_tfs: int,
        n_genes: int,
        d_latent: int = 256,
        encoder_hidden: Sequence[int] = (512, 512),
        encoder_dropout: float = 0.1,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.n_tfs    = n_tfs
        self.n_genes  = n_genes
        self.d_latent = d_latent

        self.context_encoder, self.target_encoder = build_encoders(
            n_samples=n_samples,
            d_latent=d_latent,
            hidden_dims=encoder_hidden,
            dropout=encoder_dropout,
            ema_momentum=ema_momentum,
        )

        self.W_init = nn.Parameter(
            torch.empty(n_tfs, n_genes).normal_(0, 1.0 / n_tfs ** 0.5)
        )

    def forward(self, expression: torch.Tensor, tf_mask: torch.Tensor) -> Stage1Output:
        tf_expr   = expression[tf_mask]
        s_context = self.context_encoder(tf_expr)           # (n_tfs, d_latent)

        with torch.no_grad():
            s_target = self.target_encoder(expression)      # (n_genes, d_latent)

        pred_s_target = (s_context.T @ self.W_init).T      # (n_genes, d_latent)
        loss = F.mse_loss(pred_s_target, s_target)

        return Stage1Output(
            pred_s_target=pred_s_target,
            s_target=s_target,
            s_context=s_context,
            loss=loss,
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        self.target_encoder.update_ema(self.context_encoder)

    @torch.no_grad()
    def get_W_init(self) -> torch.Tensor:
        """Return trained W_init for Stage 2 warm start."""
        return self.W_init.data.clone()

    @classmethod
    def from_network(
        cls, network, d_latent=256, encoder_hidden=(512, 512),
        encoder_dropout=0.1, ema_momentum=0.996,
    ) -> "BJEPAStage1":
        return cls(
            n_samples=network.n_samples,
            n_tfs=network.n_tfs,
            n_genes=network.n_genes,
            d_latent=d_latent,
            encoder_hidden=encoder_hidden,
            encoder_dropout=encoder_dropout,
            ema_momentum=ema_momentum,
        )


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

class BJEPAStage2(nn.Module):
    """Horseshoe sparse regression in expression space.

    For all genes simultaneously:
        pred = X_tf @ W      (n_samples, n_genes)
        L    = MSE(pred, Y) + β · KL(q(W) || p_horseshoe)

    where X_tf ∈ R^{n_samples × n_tfs} and Y ∈ R^{n_samples × n_genes}.
    """

    def __init__(
        self,
        n_tfs: int,
        n_genes: int,
        tau0: float,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.kl_weight = kl_weight

        self.horseshoe = HorseshoeRegressor(
            n_tfs=n_tfs,
            n_target_genes=n_genes,
            tau0=tau0,
        )

    def init_from_stage1(self, W_init: torch.Tensor) -> None:
        """Warm-start mu_W from Stage 1 W_init, rescaled to tau0 magnitude."""
        with torch.no_grad():
            mean_abs = W_init.abs().mean().clamp(min=1e-8)
            W_scaled = W_init * (self.horseshoe.tau0 / mean_abs)
            self.horseshoe.mu_W.copy_(W_scaled)

    def forward(
        self,
        X_tf: torch.Tensor,    # (n_samples, n_tfs)
        Y:    torch.Tensor,    # (n_samples, n_genes)
        kl_mc_samples: int = 4,
    ) -> Stage2Output:
        # Use effective horseshoe weight: mu_W * lambda_tilde * (tau / tau0)
        # This routes the regression gradient through lambda_tilde and tau,
        # breaking the mean-field VI fixed point where all lambda collapse to 1.
        # Without this, lambda only receives KL gradient → lambda* = |mu_W|/tau ≈ 1
        # for all edges → no sparsity differentiation.
        lambda_tilde = torch.exp(self.horseshoe.m_lambda)          # (n_tfs, n_genes)
        tau          = torch.exp(self.horseshoe.m_tau)             # scalar
        W_eff        = self.horseshoe.mu_W * lambda_tilde * (tau / self.horseshoe.tau0)
        pred = X_tf @ W_eff                                        # (n_samples, n_genes)

        regression_loss = F.mse_loss(pred, Y)
        kl_loss         = self.horseshoe.kl_divergence(n_mc=kl_mc_samples)
        loss            = regression_loss + self.kl_weight * kl_loss

        return Stage2Output(
            regression_loss=regression_loss,
            kl_loss=kl_loss,
            loss=loss,
        )

    @torch.no_grad()
    def edge_scores(self, gene_ids: list[str], tf_ids: list[str]) -> "pd.DataFrame":
        """Effective horseshoe weight |mu_W * lambda_tilde * tau/tau0| as ranked DataFrame."""
        import pandas as pd
        lambda_tilde = torch.exp(self.horseshoe.m_lambda)
        tau          = torch.exp(self.horseshoe.m_tau)
        W = self.horseshoe.mu_W * lambda_tilde * (tau / self.horseshoe.tau0)
        rows = [
            {"tf": tf, "target": gene, "score": W[i, j].item()}
            for i, tf in enumerate(tf_ids)
            for j, gene in enumerate(gene_ids)
            if tf != gene
        ]
        df = pd.DataFrame(rows)
        df["abs_score"] = df["score"].abs()
        return df.sort_values("abs_score", ascending=False).drop(columns="abs_score").reset_index(drop=True)

    @classmethod
    def from_network(cls, network, p0=10.0, kl_weight=1.0) -> "BJEPAStage2":
        tau0 = compute_tau0(p0=p0, n_tfs=network.n_tfs, n_samples=network.n_samples)
        return cls(
            n_tfs=network.n_tfs,
            n_genes=network.n_genes,
            tau0=tau0,
            kl_weight=kl_weight,
        )
