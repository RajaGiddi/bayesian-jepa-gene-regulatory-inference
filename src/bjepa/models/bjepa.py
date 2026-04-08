"""B-JEPA: two-stage architecture.

Stage 1 — VJEPA / BJEPA encoder training (Huang 2026, arXiv:2601.14354)
-------------------------------------------------------------------------
Implements the Variational JEPA (VJEPA) objective with the Bayesian JEPA
(B-JEPA) Product of Experts (PoE) predictor.

VJEPA replaces the deterministic I-JEPA MSE loss with a probabilistic ELBO:

    Target encoder  :  q_{θ'}(Z_T | x_T) = N(μ_q, diag(exp(log_var_q)))
                       — amortised inference distribution, EMA-updated.

    Dynamics expert :  N(μ_dyn, diag(exp(pred_log_var)))
                       where μ_dyn = (s_context.T @ W_init).T

    B-JEPA PoE      :  p_φ(Z_T | Z_C, ξ_T) = PoE(dynamics expert, N(0,I))
                       prec_poe = prec_dyn + 1
                       var_phi  = 1 / prec_poe
                       mu_phi   = var_phi * prec_dyn * mu_dyn

    ELBO            :  L = NLL(Z_T ; μ_φ, var_φ) + β · KL(q || N(0,I))
                       Z_T ~ q (reparameterisation trick)
                       NLL = 0.5 * mean(log(var_φ) + (Z_T - μ_φ)² / var_φ)
                       KL  = -0.5 * mean(1 + log_var_q - μ_q² - exp(log_var_q))

Stage 2 — Horseshoe sparse regression in expression space
----------------------------------------------------------
Freeze encoders. Use W_init as a warm start for mu_W.

Regress raw gene expression on raw TF expression with horseshoe prior:

    pred = X_tf @ W          (n_samples, n_genes)
    L    = MSE(pred, Y) + β · KL(q(W) || p_horseshoe)

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
    loss:        torch.Tensor  # scalar ELBO = NLL + β·KL
    nll:         torch.Tensor  # scalar NLL term
    kl:          torch.Tensor  # scalar KL term
    mu_q:        torch.Tensor  # (n_genes, d_latent)  inference mean
    log_var_q:   torch.Tensor  # (n_genes, d_latent)  inference log-var
    mu_phi:      torch.Tensor  # (n_genes, d_latent)  PoE predictor mean
    var_phi:     torch.Tensor  # (n_genes, d_latent)  PoE predictor var
    s_context:   torch.Tensor  # (n_tfs,  d_latent)


@dataclass
class Stage2Output:
    regression_loss: torch.Tensor  # MSE in expression space
    kl_loss:         torch.Tensor
    loss:            torch.Tensor


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class BJEPAStage1(nn.Module):
    """VJEPA + B-JEPA encoder training (Huang 2026, arXiv:2601.14354).

    Context encoder : s_context = f_θ(x_TF)                      — deterministic
    Target encoder  : (μ_q, log_var_q) = f_{θ'}(x_all)           — inference dist (EMA)
    Predictor       : [μ_dyn, log_var_dyn] = g_φ(z_c_g ‖ ξ_g)   — dynamics expert
    B-JEPA PoE      : p_φ = PoE(dynamics expert, N(0,I))
    ELBO            : NLL(Z_T ; μ_φ, var_φ) + β · KL(q ‖ N(0,I))

    Two context aggregation modes (selected by use_cross_attention):

    Mean-pool (default):
        z_c_g = mean(s_context)  — same context vector for all genes.
        Simple, fast, loses per-TF selectivity.

    Cross-attention (Ablation 3):
        z_c_g = CrossAttn(query=ξ_g, key=s_context, value=s_context)
        Each gene attends selectively to individual TF latents.
        Attention weights a_{g,d} ∈ [0,1] are interpretable as regulatory
        scores: gene g attending strongly to TF d ≈ TF d regulates gene g.
        Addresses net3 failure where mean-pool loses TF-selectivity at scale.
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
        kl_weight: float = 1.0,
        predictor_hidden: int = 256,
        use_cross_attention: bool = False,
        num_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_tfs               = n_tfs
        self.n_genes             = n_genes
        self.d_latent            = d_latent
        self.kl_weight           = kl_weight
        self.use_cross_attention = use_cross_attention

        self.context_encoder, self.target_encoder = build_encoders(
            n_samples=n_samples,
            d_latent=d_latent,
            hidden_dims=encoder_hidden,
            dropout=encoder_dropout,
            ema_momentum=ema_momentum,
        )

        # Gene-specific position tokens ξ_T — one embedding per gene.
        # In mean-pool mode: distinguishes genes in the predictor input.
        # In cross-attention mode: serves as the query for each gene.
        self.gene_embedding = nn.Embedding(n_genes, d_latent)
        nn.init.trunc_normal_(self.gene_embedding.weight, std=0.02)

        # Cross-attention: gene embeddings query TF latents.
        # Need d_latent divisible by num_attn_heads.
        if use_cross_attention:
            assert d_latent % num_attn_heads == 0, (
                f"d_latent ({d_latent}) must be divisible by num_attn_heads ({num_attn_heads})"
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_latent,
                num_heads=num_attn_heads,
                dropout=0.0,
                batch_first=True,
            )

        # Predictor MLP: (z_c_g ‖ ξ_g) → (μ_dyn ‖ log_var_dyn)
        # z_c_g is either mean-pooled or cross-attended TF context.
        self.predictor = nn.Sequential(
            nn.Linear(2 * d_latent, predictor_hidden),
            nn.GELU(),
            nn.LayerNorm(predictor_hidden),
            nn.Linear(predictor_hidden, 2 * d_latent),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        bias = self.predictor[-1].bias.data
        bias[:d_latent].zero_()       # μ_dyn bias = 0
        bias[d_latent:].fill_(-2.0)   # log_var bias = -2

    def _context_for_genes(
        self, s_context: torch.Tensor, xi_g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Aggregate TF context into per-gene vectors.

        Returns
        -------
        z_c_g      : (n_genes, d_latent)
        attn_weights: (n_genes, n_tfs) if cross-attention, else None
        """
        if self.use_cross_attention:
            # query: (1, n_genes, d_latent), key/value: (1, n_tfs, d_latent)
            q = xi_g.unsqueeze(0)           # (1, n_genes, d_latent)
            k = s_context.unsqueeze(0)      # (1, n_tfs,   d_latent)
            z_c_g, attn_w = self.cross_attn(q, k, k, need_weights=True,
                                             average_attn_weights=True)
            # z_c_g:  (1, n_genes, d_latent) → (n_genes, d_latent)
            # attn_w: (1, n_genes, n_tfs)    → (n_genes, n_tfs)
            return z_c_g.squeeze(0), attn_w.squeeze(0)
        else:
            z_c_g = s_context.mean(dim=0, keepdim=True).expand(self.n_genes, -1)
            return z_c_g, None

    def forward(self, expression: torch.Tensor, tf_mask: torch.Tensor) -> Stage1Output:
        """VJEPA ELBO + B-JEPA PoE forward pass."""
        # --- context encoder ---
        tf_expr   = expression[tf_mask]
        s_context = self.context_encoder(tf_expr)          # (n_tfs,  d_latent)

        # --- target encoder (EMA, no grad) ---
        mu_q, log_var_q = self.target_encoder(expression)  # (n_genes, d_latent)

        # --- reparameterisation sample Z_T ~ q ---
        eps = torch.randn_like(mu_q)
        Z_T = mu_q + eps * (0.5 * log_var_q).exp()

        # --- predictor: dynamics expert ---
        xi_g             = self.gene_embedding.weight      # (n_genes, d_latent)
        z_c_g, _         = self._context_for_genes(s_context, xi_g)
        pred_in          = torch.cat([z_c_g, xi_g], dim=-1)
        pred_out         = self.predictor(pred_in)
        mu_dyn, pred_log_var = pred_out.chunk(2, dim=-1)
        pred_log_var     = pred_log_var.clamp(-8.0, 4.0)

        # --- B-JEPA PoE ---
        prec_dyn = torch.exp(-pred_log_var)
        prec_poe = prec_dyn + 1.0
        var_phi  = 1.0 / prec_poe
        mu_phi   = var_phi * prec_dyn * mu_dyn

        # --- ELBO ---
        nll  = 0.5 * (var_phi.log() + (Z_T - mu_phi).pow(2) / var_phi).mean()
        kl   = -0.5 * (1.0 + log_var_q - mu_q.pow(2) - log_var_q.exp()).mean()
        loss = nll + self.kl_weight * kl

        return Stage1Output(
            loss=loss, nll=nll, kl=kl,
            mu_q=mu_q, log_var_q=log_var_q,
            mu_phi=mu_phi, var_phi=var_phi,
            s_context=s_context,
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        self.target_encoder.update_ema(self.context_encoder)

    @torch.no_grad()
    def get_W_init(
        self, expression: torch.Tensor, tf_mask: torch.Tensor
    ) -> torch.Tensor:
        """Derive W_init for Stage 2 warm-start via OLS in latent space."""
        tf_expr   = expression[tf_mask]
        s_context = self.context_encoder(tf_expr)
        mu_q, _   = self.target_encoder(expression)
        device = s_context.device
        A = s_context.T.cpu().float()
        B = mu_q.T.cpu().float()
        W = torch.linalg.lstsq(A, B).solution
        return W.to(device)

    @torch.no_grad()
    def get_attention_scores(
        self,
        expression: torch.Tensor,
        tf_mask: torch.Tensor,
        gene_ids: list[str],
        tf_ids: list[str],
    ) -> "pd.DataFrame":
        """Extract cross-attention weights as GRN edge scores.

        Only valid when use_cross_attention=True.  Attention weight a_{g,d}
        reflects how strongly gene g attends to TF d when predicting its
        latent representation — interpretable as regulatory importance.

        Returns
        -------
        DataFrame with columns [tf, target, score], sorted descending.
        """
        import pandas as pd
        assert self.use_cross_attention, (
            "get_attention_scores() requires use_cross_attention=True"
        )
        tf_expr   = expression[tf_mask]
        s_context = self.context_encoder(tf_expr)          # (n_tfs, d_latent)
        xi_g      = self.gene_embedding.weight             # (n_genes, d_latent)
        _, attn_w = self._context_for_genes(s_context, xi_g)
        # attn_w: (n_genes, n_tfs)
        attn_w = attn_w.cpu().float().numpy()

        tf_id_list   = tf_ids
        gene_id_list = gene_ids
        rows = [
            {"tf": tf_id_list[d], "target": gene_id_list[g], "score": float(attn_w[g, d])}
            for g in range(len(gene_id_list))
            for d in range(len(tf_id_list))
            if tf_id_list[d] != gene_id_list[g]
        ]
        df = pd.DataFrame(rows)
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    @classmethod
    def from_network(
        cls, network, d_latent=256, encoder_hidden=(512, 512),
        encoder_dropout=0.1, ema_momentum=0.996, kl_weight=1.0,
        predictor_hidden=256, use_cross_attention=False, num_attn_heads=4,
    ) -> "BJEPAStage1":
        return cls(
            n_samples=network.n_samples,
            n_tfs=network.n_tfs,
            n_genes=network.n_genes,
            d_latent=d_latent,
            encoder_hidden=encoder_hidden,
            encoder_dropout=encoder_dropout,
            ema_momentum=ema_momentum,
            kl_weight=kl_weight,
            predictor_hidden=predictor_hidden,
            use_cross_attention=use_cross_attention,
            num_attn_heads=num_attn_heads,
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
