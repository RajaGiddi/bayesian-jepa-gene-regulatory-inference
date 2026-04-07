"""DREAM5 data loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Map network id to human-readable name
NETWORK_NAMES = {
    1: "in_silico",
    2: "s_aureus",
    3: "e_coli",
    4: "s_cerevisiae",
}


@dataclass
class DREAM5Network:
    """Container for one DREAM5 network.

    Attributes
    ----------
    network_id:
        DREAM5 network number (1–4).
    name:
        Human-readable organism/condition name.
    expression:
        DataFrame of shape (n_samples, n_genes), log-normalised expression.
    gene_ids:
        List of gene identifiers (length n_genes).
    tf_ids:
        List of TF identifiers that are a subset of gene_ids.
    tf_mask:
        Boolean array of length n_genes; True where gene is a TF.
    gold_standard:
        DataFrame with columns [tf, target, label] for evaluated edges.
        None if no gold standard file is available.
    """

    network_id: int
    name: str
    expression: pd.DataFrame
    gene_ids: list[str]
    tf_ids: list[str]
    tf_mask: np.ndarray
    gold_standard: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.expression.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]

    @property
    def n_tfs(self) -> int:
        return len(self.tf_ids)

    @property
    def n_positive_edges(self) -> int:
        if self.gold_standard is None:
            return 0
        return int((self.gold_standard["label"] == 1).sum())

    @property
    def n_evaluated_edges(self) -> int:
        if self.gold_standard is None:
            return 0
        return len(self.gold_standard)

    @property
    def sparsity(self) -> float:
        """Fraction of TF→gene edges that are positive."""
        total = self.n_tfs * self.n_genes
        return self.n_positive_edges / total if total > 0 else float("nan")

    def tf_expression(self) -> pd.DataFrame:
        """Return expression sub-matrix for TF genes only."""
        tf_set = set(self.tf_ids)
        cols = [g for g in self.gene_ids if g in tf_set]
        return self.expression[cols]

    def target_expression(self) -> pd.DataFrame:
        """Return expression sub-matrix for non-TF genes only."""
        tf_set = set(self.tf_ids)
        cols = [g for g in self.gene_ids if g not in tf_set]
        return self.expression[cols]


def load_network(data_dir: str | Path, network_id: int) -> DREAM5Network:
    """Load a DREAM5 network from the raw data directory.

    Parameters
    ----------
    data_dir:
        Directory containing the DREAM5 TSV files (e.g. ``data/``).
    network_id:
        Network number 1–4.

    Returns
    -------
    DREAM5Network
    """
    data_dir = Path(data_dir)
    prefix = f"net{network_id}"

    # Expression matrix — first row is a header of gene IDs
    expr_path = data_dir / f"{prefix}_expression_data.tsv"
    expression = pd.read_csv(expr_path, sep="\t", index_col=None)
    # Gene IDs come from the header; use them as column names
    gene_ids = list(expression.columns)

    # TF list
    tf_path = data_dir / f"{prefix}_transcription_factors.tsv"
    tf_ids = pd.read_csv(tf_path, sep="\t", header=None)[0].tolist()

    # TF mask aligned to gene_ids
    gene_set = set(gene_ids)
    tf_set = set(tf_ids)
    tf_mask = np.array([g in tf_set for g in gene_ids], dtype=bool)

    # Gold standard (optional — not all networks have one in the public release)
    # net1/net3 list all evaluated TF→gene pairs with labels {0, 1}.
    # net2/net4 list only the positive edges (label=1); negatives are implicit
    # (all TF×gene pairs not listed).  We expand to the full evaluated set so
    # that AUROC/AUPR are well-defined.
    gs_candidates = list(data_dir.glob(f"*GoldStandard*Network{network_id}*"))
    # Exclude the signed variant (contains −1/+1 signs, not binary labels)
    gs_candidates = [p for p in gs_candidates if "Signed" not in p.name]
    gold_standard = None
    if gs_candidates:
        gs_path = gs_candidates[0]
        gs_df = pd.read_csv(gs_path, sep="\t", header=None, names=["tf", "target", "label"])

        # If only positives are listed, expand to the full TF × gene matrix
        unique_labels = gs_df["label"].unique()
        if len(unique_labels) == 1 and unique_labels[0] == 1:
            pos_set = set(zip(gs_df["tf"], gs_df["target"]))
            rows = [
                {"tf": tf, "target": gene, "label": 1 if (tf, gene) in pos_set else 0}
                for tf in tf_ids
                for gene in gene_ids
                if tf != gene  # exclude self-loops
            ]
            gs_df = pd.DataFrame(rows)

        gold_standard = gs_df

    return DREAM5Network(
        network_id=network_id,
        name=NETWORK_NAMES.get(network_id, f"network{network_id}"),
        expression=expression,
        gene_ids=gene_ids,
        tf_ids=tf_ids,
        tf_mask=tf_mask,
        gold_standard=gold_standard,
    )
