"""
metrics.py

Purpose
-------
This file contains shared evaluation functions for all models.

Why this exists
---------------
Every team member should evaluate their model using
the same metrics to ensure fair comparisons.

Metrics used in this project:
- Accuracy
- Macro F1-score
- Confusion Matrix
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Compute standard classification metrics for a model's predictions
def evaluate_classification(y_true, y_pred) -> dict[str, Any]:
    """
    Return a standard metrics dictionary used across all models.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """
    Pretty-print common classification metrics.
    """
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
