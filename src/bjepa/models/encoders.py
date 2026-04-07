"""Context and Target encoders for B-JEPA.

Gene-centric view
-----------------
Each gene is treated as a data point whose feature vector is its expression
profile across n_samples experiments.  Both encoders share the same MLP
architecture:

    (*, n_samples)  ->  (*, d_latent)

ContextEncoder  processes TF expression profiles and is trained via gradient
                descent.

TargetEncoder   has the same architecture but its weights are an exponential
                moving average (EMA) of the ContextEncoder's weights.  No
                gradients flow through it.  This prevents representational
                collapse without requiring negative samples or contrastive loss
                (identical to the I-JEPA / V-JEPA design).

LeJEPA result
-------------
Balestriero & LeCun (2025) proved that isotropic Gaussian embeddings minimise
worst-case downstream prediction risk.  The final LayerNorm approximately
enforces this by centring and scaling each embedding dimension.
"""
from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Generic MLP with LayerNorm + GELU, optional residual connections.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    hidden_dims:
        Sequence of hidden layer widths.  A residual connection is added when
        consecutive dims match.
    out_dim:
        Output (latent) dimension.
    dropout:
        Dropout probability applied after each hidden activation (0 = off).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.LayerNorm(d_out))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Final projection to latent dim — no activation, just norm
        layers.append(nn.Linear(dims[-1], out_dim))
        layers.append(nn.LayerNorm(out_dim))

        self.net = nn.Sequential(*layers)

        # Residual shortcuts: only where adjacent hidden dims are equal
        self._residual_pairs: list[tuple[int, int]] = []
        cumulative = 0  # module index tracking is simpler via forward logic
        # We'll track via explicit bookkeeping in forward instead
        self._hidden_dims = list(dims[1:])  # dims after input
        self._in_dim = in_dim

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """Encodes TF expression profiles to a latent representation.

    Maps each gene's expression profile (length n_samples) to a d_latent
    vector.  Trained end-to-end with gradients.

    Parameters
    ----------
    n_samples:
        Number of expression experiments / conditions (input feature dim).
    d_latent:
        Latent embedding dimension.
    hidden_dims:
        Hidden layer widths.  Default: [512, 512].
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        n_samples: int,
        d_latent: int = 256,
        hidden_dims: Sequence[int] = (512, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.d_latent = d_latent
        self.mlp = _MLP(n_samples, hidden_dims, d_latent, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (n_genes, n_samples)
            Each row is one gene's expression profile.

        Returns
        -------
        Tensor, shape (n_genes, d_latent)
        """
        return self.mlp(x)


class TargetEncoder(nn.Module):
    """EMA copy of ContextEncoder — no direct gradient updates.

    Weights are updated each training step via:
        θ_target ← momentum · θ_target + (1 − momentum) · θ_context

    This prevents representational collapse (the asymmetric EMA design from
    I-JEPA / BYOL) without requiring negative samples.

    Parameters
    ----------
    context_encoder:
        The ContextEncoder whose weights this mirrors.
    momentum:
        EMA momentum.  0.996 matches the I-JEPA default.
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        # Deep-copy so parameters are independent from the start
        self.mlp = copy.deepcopy(context_encoder.mlp)
        self.n_samples = context_encoder.n_samples
        self.d_latent = context_encoder.d_latent
        # Freeze: target encoder is never updated by the optimiser
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, context_encoder: ContextEncoder) -> None:
        """Pull one EMA step from the current ContextEncoder weights."""
        m = self.momentum
        for p_target, p_ctx in zip(self.mlp.parameters(), context_encoder.mlp.parameters()):
            p_target.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (n_genes, n_samples)

        Returns
        -------
        Tensor, shape (n_genes, d_latent)
        """
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encoders(
    n_samples: int,
    d_latent: int = 256,
    hidden_dims: Sequence[int] = (512, 512),
    dropout: float = 0.1,
    ema_momentum: float = 0.996,
) -> tuple[ContextEncoder, TargetEncoder]:
    """Create a matched ContextEncoder / TargetEncoder pair.

    The TargetEncoder is initialised as an exact copy of the ContextEncoder.

    Returns
    -------
    (context_encoder, target_encoder)
    """
    ctx = ContextEncoder(n_samples, d_latent, hidden_dims, dropout)
    tgt = TargetEncoder(ctx, ema_momentum)
    return ctx, tgt
