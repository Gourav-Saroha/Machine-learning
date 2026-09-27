from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

from models.data_loader import load_dataset_bundle
from models.random_forest import build_random_forest
from services.model_service import figure_to_dict, tuned_random_forest_artifacts


def feature_importance_payload(
    dataset_key: str = "iris",
    random_state: int = 42,
    n_repeats: int = 10,
    include_permutation: bool = True,
    use_tuned_model: bool = True,
) -> dict[str, Any]:
    if n_repeats < 1 or n_repeats > 50:
        raise ValueError("n_repeats must be between 1 and 50.")

    if use_tuned_model:
        artifacts = tuned_random_forest_artifacts(dataset_key, random_state)
        model = artifacts["model"]
        bundle = artifacts["bundle"]
    else:
        bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=42)
        model = build_random_forest(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=random_state,
            oob_score=True,
        )
        model.fit(bundle.X_train, bundle.y_train)

    gini_df = pd.DataFrame(
        {
            "feature": bundle.feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    permutation_df = pd.DataFrame(columns=["feature", "importance", "std"])
    permutation_figure_dict: dict[str, Any] | None = None

    if include_permutation:
        permutation = permutation_importance(
            model,
            bundle.X_test,
            bundle.y_test,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        permutation_df = pd.DataFrame(
            {
                "feature": bundle.feature_names,
                "importance": permutation.importances_mean,
                "std": permutation.importances_std,
            }
        ).sort_values("importance", ascending=False)

    gini_figure = go.Figure(
        go.Bar(
            x=gini_df["importance"].round(6),
            y=gini_df["feature"],
            orientation="h",
            marker=dict(color="#0f766e"),
        )
    )
    gini_figure.update_layout(
        title="Gini / MDI Feature Importance",
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Importance",
        yaxis_title="Feature",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    if include_permutation:
        permutation_figure = go.Figure(
            go.Bar(
                x=permutation_df["importance"].round(6),
                y=permutation_df["feature"],
                orientation="h",
                error_x=dict(type="data", array=permutation_df["std"].round(6)),
                marker=dict(color="#2563eb"),
            )
        )
        permutation_figure.update_layout(
            title="Permutation Importance on Held-Out Test Data",
            margin=dict(l=40, r=20, t=55, b=40),
            xaxis_title="Mean importance decrease",
            yaxis_title="Feature",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        permutation_figure_dict = figure_to_dict(permutation_figure)

    return {
        "has_permutation": include_permutation,
        "gini_rows": gini_df.round(6).to_dict(orient="records"),
        "permutation_rows": permutation_df.round(6).to_dict(orient="records"),
        "charts": {
            "gini": figure_to_dict(gini_figure),
            "permutation": permutation_figure_dict,
        },
    }
