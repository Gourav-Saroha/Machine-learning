from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def classification_metrics(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    labels: list[str],
    cv: int = 5,
) -> dict[str, Any]:
    """Evaluate a fitted classifier on the shared split."""
    y_pred = model.predict(X_test)
    cv_model = clone(model)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="accuracy")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)

    table_rows = []
    for label in labels:
        row = report[label]
        table_rows.append(
            {
                "label": label,
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "f1_score": round(float(row["f1-score"]), 4),
                "support": int(row["support"]),
            }
        )

    weighted = report["weighted avg"]
    return {
        "train_accuracy": round(float(model.score(X_train, y_train)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "cv_accuracy": round(float(np.mean(cv_scores)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_precision": round(float(weighted["precision"]), 4),
        "weighted_recall": round(float(weighted["recall"]), 4),
        "weighted_f1_score": round(float(weighted["f1-score"]), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_rows": table_rows,
        "prediction_distribution": (
            np.bincount(y_pred, minlength=len(labels)).astype(int).tolist()
        ),
    }
