"""GENIE3: Gene Network Inference with Ensemble of trees.

Reference:
    Huynh-Thu et al. (2010). Inferring Regulatory Networks from Expression
    Data Using Tree-Based Methods. PLOS ONE.

For each target gene j, an Extra-Trees regressor is trained to predict j's
expression from the TF expression profiles. Feature importances serve as
directed edge weight proxies: importance(TF_i → gene_j) = how much TF_i
contributes to predicting gene_j.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel, delayed
from tqdm import tqdm

from bjepa.data.dream5 import DREAM5Network


def _fit_one_gene(
    target_col: str,
    X_tf: np.ndarray,
    y: np.ndarray,
    tf_ids: list[str],
    n_estimators: int,
    max_features: str | float,
    random_state: int,
) -> pd.DataFrame:
    """Fit one Extra-Trees model and return its importance scores as a DataFrame."""
    reg = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
    )
    reg.fit(X_tf, y)

    importances = reg.feature_importances_
    # Normalise so scores sum to 1 across TFs for this target gene.
    # Prevents high-variance genes from dominating the global ranking.
    total = importances.sum()
    if total > 0:
        importances = importances / total

    return pd.DataFrame({
        "tf": tf_ids,
        "target": target_col,
        "score": importances,
    })


class GENIE3:
    """GENIE3 GRN inference via Extra-Trees feature importances.

    Parameters
    ----------
    n_estimators:
        Number of trees per regressor (original paper uses 1000; 500 is
        faster with minimal accuracy loss).
    max_features:
        Features considered at each split. "sqrt" matches the GENIE3 paper
        default for classification-style splits; can also pass a float (e.g.
        0.1 for 10% of TFs).
    n_jobs:
        Parallel jobs across target genes. -1 = all cores.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_features: str | float = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.predictions_: pd.DataFrame | None = None

    def fit_predict(self, network: DREAM5Network) -> pd.DataFrame:
        """Run GENIE3 on a DREAM5Network and return a scored edge list.

        Parameters
        ----------
        network:
            Loaded DREAM5Network instance.

        Returns
        -------
        DataFrame with columns [tf, target, score], sorted by score descending.
        One row per (TF, target-gene) pair; TF→TF edges included (consistent
        with DREAM5 evaluation convention).
        """
        tf_set = set(network.tf_ids)

        # TF feature matrix: shape (n_samples, n_tfs)
        tf_cols = [g for g in network.gene_ids if g in tf_set]
        X_tf = network.expression[tf_cols].values.astype(np.float32)

        # All genes are prediction targets (including TFs themselves)
        target_genes = network.gene_ids

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_gene)(
                target_col=gene,
                X_tf=X_tf,
                y=network.expression[gene].values.astype(np.float32),
                tf_ids=tf_cols,
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                random_state=self.random_state,
            )
            for gene in tqdm(target_genes, desc=f"GENIE3 net{network.network_id}", unit="gene")
        )

        predictions = pd.concat(results, ignore_index=True)
        # Self-regulation edges (TF predicting itself) are spuriously high;
        # zero them out to match DREAM5 convention.
        self_mask = predictions["tf"] == predictions["target"]
        predictions.loc[self_mask, "score"] = 0.0

        predictions = predictions.sort_values("score", ascending=False).reset_index(drop=True)
        self.predictions_ = predictions
        return predictions
