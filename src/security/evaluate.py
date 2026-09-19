"""Binary detection, category breakdown, and probability diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support

from src.security.risk import threshold_risk


def evaluate_binary(labels, risk, threshold=.5):
    prediction = threshold_risk(risk, threshold)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, prediction)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0),
            "confusion_matrix": matrix.tolist(), "brier_score": float(brier_score_loss(labels, risk))}


def category_breakdown(frame, risk, threshold=.5):
    predicted = threshold_risk(risk, threshold)
    result = {}
    for category, indices in frame.groupby("attack_cat").groups.items():
        subset = predicted[list(indices)]
        result[str(category)] = {"count": int(len(indices)), "mean_risk": float(np.mean(np.asarray(risk)[list(indices)])),
                                 "attack_prediction_rate": float(np.mean(subset))}
    return result
