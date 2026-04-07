"""Evaluation metrics for GRN inference (AUROC, AUPR).

Following DREAM5 convention: predictions are a scored edge list; only edges
present in the gold standard are counted; ties are broken by averaging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


def evaluate_predictions(
    predictions: pd.DataFrame,
    gold_standard: pd.DataFrame,
) -> dict[str, float]:
    """Compute AUROC and AUPR against a DREAM5 gold standard.

    Parameters
    ----------
    predictions:
        DataFrame with columns [tf, target, score]. Higher score = more
        likely regulatory edge. All TF→gene pairs should be included.
    gold_standard:
        DataFrame with columns [tf, target, label] (label ∈ {0, 1}).

    Returns
    -------
    dict with keys 'auroc' and 'aupr'.
    """
    merged = gold_standard.merge(predictions, on=["tf", "target"], how="left")
    merged["score"] = merged["score"].fillna(0.0)

    labels = merged["label"].values
    scores = merged["score"].values

    return {
        "auroc": auroc(labels, scores),
        "aupr": aupr(labels, scores),
    }
