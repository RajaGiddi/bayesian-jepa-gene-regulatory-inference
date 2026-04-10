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
from .kan import BSplineActivation


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class Stage1Output:
    loss:        torch.Tensor  # total: α·L_fwd + (1-α)·L_bwd
    nll:         torch.Tensor  # forward NLL
    kl:          torch.Tensor  # forward KL
    mu_q:        torch.Tensor  # (n_genes, d_latent)  inference mean
    log_var_q:   torch.Tensor  # (n_genes, d_latent)  inference log-var
    mu_phi:      torch.Tensor  # (n_genes, d_latent)  PoE predictor mean
    var_phi:     torch.Tensor  # (n_genes, d_latent)  PoE predictor var
    s_context:   torch.Tensor  # (n_tfs,  d_latent)
    nll_bwd:     torch.Tensor | None = None  # backward NLL (None if unidirectional)
    kl_bwd:      torch.Tensor | None = None  # backward KL  (None if unidirectional)


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
        use_bidirectional: bool = False,
        fwd_weight: float = 0.7,
        use_stochastic_mask: bool = False,
        tf_bias: float = 0.7,
    ) -> None:
        super().__init__()
        self.n_tfs                = n_tfs
        self.n_genes              = n_genes
        self.d_latent             = d_latent
        self.kl_weight            = kl_weight
        self.use_cross_attention  = use_cross_attention
        self.use_bidirectional    = use_bidirectional
        self.fwd_weight           = fwd_weight   # α: weight on forward ELBO
        self.use_stochastic_mask  = use_stochastic_mask
        self.tf_bias              = tf_bias      # p(context | TF); 1.0 = standard B-JEPA

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

        # Forward predictor MLP: (z_c_g ‖ ξ_g) → (μ_dyn ‖ log_var_dyn)
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

        # Backward predictor (BiJEPA): non-TF context → predict TF representations.
        # Mirrors the forward predictor structure exactly.
        # Only instantiated when use_bidirectional=True.
        if use_bidirectional:
            # TF-specific position tokens ξ_T_back — one per TF (backward targets).
            self.tf_embedding = nn.Embedding(n_tfs, d_latent)
            nn.init.trunc_normal_(self.tf_embedding.weight, std=0.02)

            self.bwd_predictor = nn.Sequential(
                nn.Linear(2 * d_latent, predictor_hidden),
                nn.GELU(),
                nn.LayerNorm(predictor_hidden),
                nn.Linear(predictor_hidden, 2 * d_latent),
            )
            nn.init.zeros_(self.bwd_predictor[-1].weight)
            bwd_bias = self.bwd_predictor[-1].bias.data
            bwd_bias[:d_latent].zero_()
            bwd_bias[d_latent:].fill_(-2.0)

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
            n_tgt = xi_g.shape[0]
            z_c_g = s_context.mean(dim=0, keepdim=True).expand(n_tgt, -1)
            return z_c_g, None

    def _sample_context_mask(self, tf_mask: torch.Tensor) -> torch.Tensor:
        """Sample a stochastic context mask with TF-biased probability.

        Each gene is independently assigned to context with probability:
          p = tf_bias      if the gene is a TF
          p = 1 - tf_bias  if the gene is a non-TF

        tf_bias=1.0 → TFs always context (reduces to standard B-JEPA).
        tf_bias=0.5 → fully symmetric, no causal prior.

        Guarantees at least 1 context gene and 1 target gene.
        """
        probs        = torch.where(tf_mask,
                                   tf_mask.new_full(tf_mask.shape, self.tf_bias, dtype=torch.float),
                                   tf_mask.new_full(tf_mask.shape, 1.0 - self.tf_bias, dtype=torch.float))
        mask         = torch.bernoulli(probs).bool()
        # Safety: ensure at least 1 context and 1 target
        if mask.all():
            idx      = mask.nonzero(as_tuple=True)[0][torch.randint(mask.sum(), (1,))]
            mask[idx] = False
        if not mask.any():
            idx      = (~mask).nonzero(as_tuple=True)[0][torch.randint((~mask).sum(), (1,))]
            mask[idx] = True
        return mask

    def _bwd_elbo(
        self, expression: torch.Tensor, tf_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward ELBO: non-TF gene context → predict TF representations.

        Follows the same VJEPA + B-JEPA PoE formulation as the forward pass
        but with directions swapped (BiJEPA §3.2):
          - Context encoder processes non-TF genes  → s_ctx_bwd
          - Target encoder (EMA) processes TF genes → q(Z_TF)
          - Backward predictor: mean-pool(s_ctx_bwd) ‖ ξ_t → dynamics expert

        Returns (L_bwd, nll_bwd, kl_bwd).
        """
        # Context: non-TF genes → latents via shared context encoder
        nontf_expr  = expression[~tf_mask]                        # (n_nontf, n_samples)
        s_ctx_bwd   = self.context_encoder(nontf_expr)            # (n_nontf, d_latent)

        # Target: TF genes → inference distribution via shared EMA target encoder
        tf_expr     = expression[tf_mask]                         # (n_tfs, n_samples)
        mu_q_tf, log_var_q_tf = self.target_encoder(tf_expr)     # (n_tfs, d_latent)

        # Reparameterisation: Z_TF ~ q(Z_TF)
        eps_tf = torch.randn_like(mu_q_tf)
        Z_TF   = mu_q_tf + eps_tf * (0.5 * log_var_q_tf).exp()

        # Backward predictor: mean-pooled non-TF context + TF embedding → dynamics expert
        z_c_bwd = s_ctx_bwd.mean(dim=0, keepdim=True).expand(self.n_tfs, -1)  # (n_tfs, d)
        xi_t    = self.tf_embedding.weight                                      # (n_tfs, d)
        pred_in_bwd          = torch.cat([z_c_bwd, xi_t], dim=-1)
        pred_out_bwd         = self.bwd_predictor(pred_in_bwd)
        mu_dyn_bwd, plv_bwd  = pred_out_bwd.chunk(2, dim=-1)
        plv_bwd              = plv_bwd.clamp(-8.0, 4.0)

        # B-JEPA PoE (backward)
        prec_dyn_bwd = torch.exp(-plv_bwd)
        prec_poe_bwd = prec_dyn_bwd + 1.0
        var_phi_bwd  = 1.0 / prec_poe_bwd
        mu_phi_bwd   = var_phi_bwd * prec_dyn_bwd * mu_dyn_bwd

        # Backward ELBO
        nll_bwd = 0.5 * (
            var_phi_bwd.log() + (Z_TF - mu_phi_bwd).pow(2) / var_phi_bwd
        ).mean()
        kl_bwd  = -0.5 * (
            1.0 + log_var_q_tf - mu_q_tf.pow(2) - log_var_q_tf.exp()
        ).mean()

        return nll_bwd + self.kl_weight * kl_bwd, nll_bwd, kl_bwd

    def forward(self, expression: torch.Tensor, tf_mask: torch.Tensor) -> Stage1Output:
        """VJEPA ELBO + B-JEPA PoE forward pass.

        When use_bidirectional=True, also computes the backward ELBO
        (non-TF context → predict TF representations) and combines:

            loss = fwd_weight · L_fwd + (1 - fwd_weight) · L_bwd

        When use_stochastic_mask=True (SM-JEPA), context/target roles are
        sampled stochastically each step with TF-biased probability tf_bias:
          - TF gene   → context with p = tf_bias
          - non-TF    → context with p = 1 − tf_bias
        This avoids a separate backward predictor: bidirectional consistency
        emerges as implicit regularisation on the shared encoder. At inference
        (eval mode or get_W_init), the standard tf_mask is always used.
        """
        # --- context mask: stochastic (train) or fixed TF mask (eval/standard) ---
        if self.use_stochastic_mask and self.training:
            context_mask = self._sample_context_mask(tf_mask)
        else:
            context_mask = tf_mask

        target_ids = (~context_mask).nonzero(as_tuple=True)[0]  # indices of target genes

        # --- context encoder ---
        s_context = self.context_encoder(expression[context_mask])   # (n_ctx, d_latent)

        # --- target encoder (EMA, no grad) — only target genes ---
        mu_q, log_var_q = self.target_encoder(expression[~context_mask])  # (n_tgt, d_latent)

        # --- reparameterisation sample Z_T ~ q ---
        eps = torch.randn_like(mu_q)
        Z_T = mu_q + eps * (0.5 * log_var_q).exp()

        # --- predictor: dynamics expert ---
        xi_g             = self.gene_embedding(target_ids)         # (n_tgt, d_latent)
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

        # --- forward ELBO ---
        nll  = 0.5 * (var_phi.log() + (Z_T - mu_phi).pow(2) / var_phi).mean()
        kl   = -0.5 * (1.0 + log_var_q - mu_q.pow(2) - log_var_q.exp()).mean()
        l_fwd = nll + self.kl_weight * kl

        # --- backward ELBO (BiJEPA) ---
        nll_bwd = kl_bwd = None
        if self.use_bidirectional:
            l_bwd, nll_bwd, kl_bwd = self._bwd_elbo(expression, tf_mask)
            loss = self.fwd_weight * l_fwd + (1.0 - self.fwd_weight) * l_bwd
        else:
            loss = l_fwd

        return Stage1Output(
            loss=loss, nll=nll, kl=kl,
            mu_q=mu_q, log_var_q=log_var_q,
            mu_phi=mu_phi, var_phi=var_phi,
            s_context=s_context,
            nll_bwd=nll_bwd, kl_bwd=kl_bwd,
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
        use_bidirectional=False, fwd_weight=0.7,
        use_stochastic_mask=False, tf_bias=0.7,
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
            use_bidirectional=use_bidirectional,
            fwd_weight=fwd_weight,
            use_stochastic_mask=use_stochastic_mask,
            tf_bias=tf_bias,
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


# ---------------------------------------------------------------------------
# KAN Stage 2
# ---------------------------------------------------------------------------

class KANStage2(nn.Module):
    """Factored KAN regression for GRN inference.

    Replaces the linear Stage 2 regression with a two-part model:

        H   = KAN_act(X_tf)    — per-TF learnable B-spline activation
        Y_g = H @ W_eff        — horseshoe-regularised linear mixing

    Biological motivation
    ---------------------
    Gene regulation follows nonlinear dose-response kinetics (Hill equation).
    A TF at low concentration may have no effect; at high concentration it
    saturates.  The B-spline φ_t(x_t) learns this curve from data.

    The factored design keeps the parameter count near-identical to the linear
    Stage 2 (one spline per TF, not one per TF-gene pair):

        Linear horseshoe   :  D × G weights
        Factored KAN       :  D × (G_grid + k) spline coefs  +  D × G weights

    For net2 (D=99, G=2810, grid G=5, k=3): 278K → 279K parameters.

    Score
    -----
        score(TF_t → gene_g) = |W_eff[t,g]| × ||φ_t||_range

    The spline amplitude weights the linear score by how much TF_t's
    nonlinear activation actually varies over observed expression levels.
    A flat spline (φ_t ≈ const) contributes negligible signal regardless
    of W_eff magnitude.

    Parameters
    ----------
    n_tfs, n_genes : network dimensions
    tau0           : horseshoe global scale (from compute_tau0)
    kl_weight      : ELBO β for KL loss term
    grid_size      : G — number of B-spline intervals (G+k basis functions)
    spline_order   : k — polynomial order (k=3 → cubic splines)
    """

    def __init__(
        self,
        n_tfs:        int,
        n_genes:      int,
        tau0:         float,
        kl_weight:    float = 1.0,
        grid_size:    int   = 5,
        spline_order: int   = 3,
    ) -> None:
        super().__init__()
        self.kl_weight = kl_weight

        self.kan_act  = BSplineActivation(
            n_inputs=n_tfs,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.horseshoe = HorseshoeRegressor(
            n_tfs=n_tfs,
            n_target_genes=n_genes,
            tau0=tau0,
        )

    def init_from_stage1(self, W_init: torch.Tensor) -> None:
        """Warm-start from Stage 1 W_init.

        Initialises the linear mixing weights from W_init (same as BJEPAStage2).
        KAN activations start as exact identity (w_s=0, coef=0) so initial
        behaviour is KAN(X_tf) = X_tf — perfectly aligned with W_init.
        Splines then learn nonlinear corrections from this linear baseline.
        """
        with torch.no_grad():
            mean_abs = W_init.abs().mean().clamp(min=1e-8)
            W_scaled = W_init * (self.horseshoe.tau0 / mean_abs)
            self.horseshoe.mu_W.copy_(W_scaled)
            # Identity initialisation — spline corrections start at zero
            self.kan_act.w_s.fill_(0.0)
            nn.init.zeros_(self.kan_act.coef)

    def forward(
        self,
        X_tf: torch.Tensor,     # (n_samples, n_tfs)
        Y:    torch.Tensor,     # (n_samples, n_genes)
        kl_mc_samples: int = 4,
    ) -> Stage2Output:
        # Apply per-TF spline activations
        H = self.kan_act(X_tf)                                 # (n, D)

        # Horseshoe linear mixing (same routing as BJEPAStage2)
        lambda_tilde = torch.exp(self.horseshoe.m_lambda)      # (D, G)
        tau          = torch.exp(self.horseshoe.m_tau)         # scalar
        W_eff        = self.horseshoe.mu_W * lambda_tilde * (tau / self.horseshoe.tau0)

        pred = H @ W_eff                                        # (n, G)

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
        """Edge scores accounting for both linear weight and spline amplitude.

        score(TF_t → gene_g) = |W_eff[t,g]| × ||φ_t||_range

        The spline amplitude ||φ_t||_range is the max |φ_t(x)| over the
        observed input range, capturing how much TF_t's nonlinear activation
        actually varies.  A flat activation → near-zero amplitude → suppressed
        score regardless of W_eff magnitude.
        """
        import pandas as pd

        lambda_tilde = torch.exp(self.horseshoe.m_lambda)
        tau          = torch.exp(self.horseshoe.m_tau)
        W = self.horseshoe.mu_W * lambda_tilde * (tau / self.horseshoe.tau0)  # (D, G)

        # Per-TF spline amplitude
        amp = self.kan_act.spline_amplitude()  # (D,)

        # Combined: |W_eff[t,g]| × amp[t]
        scores = W.abs() * amp.unsqueeze(1)    # (D, G)

        rows = [
            {"tf": tf, "target": gene, "score": scores[i, j].item()}
            for i, tf in enumerate(tf_ids)
            for j, gene in enumerate(gene_ids)
            if tf != gene
        ]
        df = pd.DataFrame(rows)
        df["abs_score"] = df["score"].abs()
        return (df.sort_values("abs_score", ascending=False)
                  .drop(columns="abs_score")
                  .reset_index(drop=True))

    @classmethod
    def from_network(
        cls, network, p0=10.0, kl_weight=1.0,
        grid_size=5, spline_order=3,
    ) -> "KANStage2":
        tau0 = compute_tau0(p0=p0, n_tfs=network.n_tfs, n_samples=network.n_samples)
        return cls(
            n_tfs=network.n_tfs,
            n_genes=network.n_genes,
            tau0=tau0,
            kl_weight=kl_weight,
            grid_size=grid_size,
            spline_order=spline_order,
        )
