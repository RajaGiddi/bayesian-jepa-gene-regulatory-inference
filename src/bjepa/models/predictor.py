"""JEPA Predictor for B-JEPA.

Role
----
The predictor takes the context encoder's TF latent representations and maps
them to predicted latent representations for each target gene, conditioned on
regulatory graph structure.

Analogy to I-JEPA
-----------------
In I-JEPA the predictor maps:
    (context embedding) + (masked patch positions) --> predicted patch embeddings

In B-JEPA the predictor maps:
    (TF latent embeddings) + (regulatory adjacency weights W) --> predicted target gene embeddings

The "positions" in I-JEPA become the regulatory edge weights W ∈ R^{n_tfs × n_target_genes}
— which TFs are hypothesised to regulate which genes, and with what strength.

Architecture
------------
For each target gene g, the predictor computes a weighted aggregation of TF
latents, where the weights come from the regulatory weight vector W[:, g]:

    agg_g = sum_i  W[i, g] * s_context[i]          (n_tfs, d_latent) -> (d_latent,)
    pred_s_target[g] = MLP_head(agg_g)              (d_latent,) -> (d_latent,)

This is deliberately narrow (fewer layers, smaller hidden dim than the encoder)
following I-JEPA's design principle: the predictor should learn abstract
mappings, not memorise identity functions.

The regulatory weights W are the horseshoe-prior quantities — they will be
provided by the HorseshoeRegressor and carry posterior uncertainty.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """Predicts target gene latents from TF latents + regulatory weights.

    Parameters
    ----------
    d_latent:
        Latent dimension (must match encoder output).
    hidden_dim:
        Hidden width of the predictor MLP head.  Kept narrower than the
        encoder (I-JEPA design principle).
    n_layers:
        Number of hidden layers in the MLP head.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_latent: int = 256,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent

        # Narrow MLP head: d_latent -> hidden_dim -> ... -> d_latent
        layers: list[nn.Module] = []
        in_d = d_latent
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers += [nn.Linear(hidden_dim, d_latent), nn.LayerNorm(d_latent)]
        self.head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        s_context: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target gene latents via regulatory-weighted aggregation.

        Parameters
        ----------
        s_context : Tensor, shape (n_tfs, d_latent)
            TF latent representations from the ContextEncoder.
        W : Tensor, shape (n_tfs, n_target_genes)
            Regulatory weight matrix.  Entry W[i, g] is the (posterior mean)
            regulatory influence of TF i on target gene g.  Provided by the
            HorseshoeRegressor.  Does NOT need to be sparse at this stage —
            the horseshoe prior will push most weights toward zero during
            training.

        Returns
        -------
        Tensor, shape (n_target_genes, d_latent)
            Predicted latent representations for each target gene.
        """
        # Weighted sum of TF latents for each target gene
        # s_context: (n_tfs, d_latent)
        # W:         (n_tfs, n_target_genes)
        # agg:       (n_target_genes, d_latent)
        agg = W.T @ s_context  # (n_target_genes, d_latent)

        # Normalise by the number of TFs to keep scale stable regardless of
        # how many TFs are active regulators for a given gene
        agg = agg / (s_context.shape[0] ** 0.5)

        return self.head(agg)
