from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from models.data_loader import load_dataset_bundle
from models.random_forest import build_random_forest
from services.model_service import figure_to_dict, tuned_random_forest_artifacts


def oob_payload(
    dataset_key: str = "iris",
    random_state: int = 42,
    max_estimators: int = 200,
    use_tuned_model: bool = True,
) -> dict[str, Any]:
    if random_state < 0:
        raise ValueError("random_state must be non-negative.")
    if max_estimators < 10 or max_estimators > 500:
        raise ValueError("max_estimators must be between 10 and 500.")

    if use_tuned_model:
        tuned = tuned_random_forest_artifacts(dataset_key, random_state)
        model = tuned["model"]
        metrics = tuned["metrics"]
    else:
        bundle = load_dataset_bundle(dataset_key=dataset_key)
        model = build_random_forest(
            n_estimators=min(max_estimators, 100),
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=random_state,
            oob_score=True,
        )
        model.fit(bundle.X_train, bundle.y_train)
        metrics = {
            "oob_score": round(float(model.oob_score_), 4),
            "oob_error": round(float(1 - model.oob_score_), 4),
        }

    bundle = load_dataset_bundle(dataset_key=dataset_key)

    points = sorted(set([10, 25, 50, 75, 100, 150, max_estimators]))
    scores = []
    for point in points:
        rf = build_random_forest(
            n_estimators=point,
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            min_samples_leaf=model.min_samples_leaf,
            max_features=model.max_features,
            random_state=random_state,
            oob_score=True,
        )
        rf.fit(bundle.X_train, bundle.y_train)
        scores.append(round(float(rf.oob_score_), 4))

    figure = go.Figure(
        go.Scatter(
            x=points,
            y=scores,
            mode="lines+markers",
            line=dict(color="#0f766e", width=3),
            marker=dict(size=9, color="#f97316"),
        )
    )
    figure.update_layout(
        title="OOB Score vs Number of Trees",
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Number of trees",
        yaxis_title="OOB score",
        yaxis_range=[0, 1.05],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return {
        "oob_score": metrics["oob_score"],
        "oob_error": metrics["oob_error"],
        "n_estimators": int(model.n_estimators),
        "max_depth": model.max_depth,
        "max_features": model.max_features,
        "charts": {
            "oob_curve": figure_to_dict(figure),
        },
    }
