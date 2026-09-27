from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.base import clone

from models.bagging import build_bagging, tune_bagging
from models.data_loader import dataset_summary, load_dataset_bundle
from models.decision_tree import build_decision_tree, export_rules, tune_decision_tree
from models.evaluation import classification_metrics
from models.random_forest import build_random_forest, tune_random_forest


DEFAULT_SPLIT_RANDOM_STATE = 42


def figure_to_dict(figure: go.Figure) -> dict[str, Any]:
    return json.loads(pio.to_json(figure))


def _curve_points(max_count: int) -> list[int]:
    if max_count <= 10:
        return list(range(1, max_count + 1))
    points = {1, 5, 10, max_count}
    for value in np.linspace(1, max_count, num=min(10, max_count), dtype=int):
        points.add(int(value))
    return sorted(points)


def _validate_int(name: str, value: int, minimum: int, maximum: int | None = None) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")


def _validate_float(name: str, value: float, minimum: float, maximum: float) -> None:
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def _confusion_matrix_figure(matrix: list[list[int]], labels: list[str], title: str) -> dict[str, Any]:
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="YlGnBu",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Predicted label",
        yaxis_title="True label",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure_to_dict(figure)


def _feature_importance_figure(feature_names: list[str], values: list[float], title: str) -> dict[str, Any]:
    ordered = sorted(zip(feature_names, values), key=lambda item: item[1])
    figure = go.Figure(
        go.Bar(
            x=[round(value, 6) for _, value in ordered],
            y=[name for name, _ in ordered],
            orientation="h",
            marker=dict(color=[value for _, value in ordered], colorscale="Tealgrn"),
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Importance",
        yaxis_title="Feature",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure_to_dict(figure)


def _metric_snapshot_figure(metrics: dict[str, Any], title: str) -> dict[str, Any]:
    keys = ["train_accuracy", "cv_accuracy", "test_accuracy", "precision", "recall", "f1_score"]
    labels = ["Train", "CV", "Test", "Precision", "Recall", "F1"]
    values = [metrics[key] for key in keys]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=["#0f766e", "#14b8a6", "#0ea5e9", "#2563eb", "#f59e0b", "#f97316"]),
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=55, b=40),
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure_to_dict(figure)


def _prediction_distribution_figure(prediction_distribution: list[int], labels: list[str], title: str) -> dict[str, Any]:
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=prediction_distribution,
            marker=dict(color=["#1d4ed8", "#0f766e", "#f97316"][: len(labels)]),
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=55, b=40),
        yaxis_title="Predicted count",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure_to_dict(figure)


def _accuracy_curve_figure(x_values: list[int], y_values: list[float], title: str, x_label: str) -> dict[str, Any]:
    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=9, color="#f97316"),
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title=x_label,
        yaxis_title="Test accuracy",
        yaxis_range=[0, 1.05],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure_to_dict(figure)


def _probability_figure(labels: list[str], probabilities: list[float], title: str) -> dict[str, Any]:
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=probabilities,
            marker=dict(color=["#2563eb", "#14b8a6", "#f97316"][: len(labels)]),
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=55, b=40),
        yaxis_title="Probability",
        yaxis_range=[0, 1.05],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure_to_dict(figure)


def _feature_defaults(bundle) -> dict[str, float]:
    return {
        feature_name: round(float(bundle.X_train[feature_name].mean()), 3)
        for feature_name in bundle.feature_names
    }


def _decision_path_details(model, sample_row, feature_names: list[str], target_names: list[str]) -> list[dict[str, Any]]:
    node_indicator = model.decision_path(sample_row.to_frame().T)
    leaf_id = model.apply(sample_row.to_frame().T)[0]
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    details = []
    node_indices = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    for node_id in node_indices:
        if node_id == leaf_id or feature[node_id] == -2:
            details.append(
                {
                    "node_id": int(node_id),
                    "rule": "Leaf node reached",
                    "feature": None,
                    "threshold": None,
                }
            )
            continue
        feature_name = feature_names[feature[node_id]]
        sample_value = float(sample_row.iloc[feature[node_id]])
        comparator = "<=" if sample_value <= threshold[node_id] else ">"
        details.append(
            {
                "node_id": int(node_id),
                "feature": feature_name,
                "threshold": round(float(threshold[node_id]), 4),
                "sample_value": round(sample_value, 4),
                "rule": f"{feature_name} {comparator} {threshold[node_id]:.4f}",
            }
        )
    return details


def _decision_boundary_figure(model, bundle, feature_x: str, feature_y: str, title: str) -> dict[str, Any]:
    x_range = np.linspace(bundle.X[feature_x].min() - 0.5, bundle.X[feature_x].max() + 0.5, 140)
    y_range = np.linspace(bundle.X[feature_y].min() - 0.5, bundle.X[feature_y].max() + 0.5, 140)
    xx, yy = np.meshgrid(x_range, y_range)

    grid = np.tile(bundle.X_train.mean().to_numpy(), (xx.size, 1))
    feature_x_index = bundle.feature_names.index(feature_x)
    feature_y_index = bundle.feature_names.index(feature_y)
    grid[:, feature_x_index] = xx.ravel()
    grid[:, feature_y_index] = yy.ravel()
    grid_frame = bundle.X_train.iloc[: xx.size].copy()
    if len(grid_frame) != xx.size:
        repeats = int(np.ceil(xx.size / len(bundle.X_train)))
        grid_frame = np.tile(bundle.X_train.mean().to_numpy(), (xx.size, 1))
        import pandas as pd

        grid_frame = pd.DataFrame(grid_frame, columns=bundle.feature_names)
    else:
        grid_frame.loc[:, :] = bundle.X_train.mean().to_numpy()
    grid_frame.iloc[:, feature_x_index] = xx.ravel()
    grid_frame.iloc[:, feature_y_index] = yy.ravel()
    predictions = model.predict(grid_frame).reshape(xx.shape)

    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=predictions,
            colorscale=[
                [0.0, "rgba(37, 99, 235, 0.18)"],
                [0.33, "rgba(37, 99, 235, 0.18)"],
                [0.34, "rgba(15, 118, 110, 0.16)"],
                [0.66, "rgba(15, 118, 110, 0.16)"],
                [0.67, "rgba(249, 115, 22, 0.16)"],
                [1.0, "rgba(249, 115, 22, 0.16)"],
            ],
            showscale=False,
            hoverinfo="skip",
            contours=dict(showlines=False),
        )
    )
    palette = ["#2563eb", "#0f766e", "#f97316"]
    for class_index, class_name in enumerate(bundle.target_names):
        class_mask = bundle.y_test == class_index
        figure.add_trace(
            go.Scatter(
                x=bundle.X_test.loc[class_mask, feature_x],
                y=bundle.X_test.loc[class_mask, feature_y],
                mode="markers",
                name=class_name,
                marker=dict(size=8, color=palette[class_index], line=dict(color="white", width=1)),
            )
        )
    figure.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return figure_to_dict(figure)


def _feature_explanation_rows(feature_names: list[str], importances: list[float], top_k: int = 3) -> list[dict[str, Any]]:
    rows = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {"feature": feature_name, "importance": round(float(importance), 4)}
        for feature_name, importance in rows
    ]


@lru_cache(maxsize=64)
def _decision_tree_artifacts(dataset_key: str, max_depth: int | None, min_samples_leaf: int, random_state: int) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    model = build_decision_tree(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    model.fit(bundle.X_train, bundle.y_train)
    metrics = classification_metrics(model, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test, bundle.target_names)
    return {"model": model, "bundle": bundle, "metrics": metrics}


@lru_cache(maxsize=16)
def _tuned_decision_tree_artifacts(dataset_key: str, random_state: int) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    grid = tune_decision_tree(bundle.X_train, bundle.y_train, random_state=random_state)
    model = grid.best_estimator_
    metrics = classification_metrics(model, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test, bundle.target_names)
    return {
        "model": model,
        "bundle": bundle,
        "metrics": metrics,
        "best_params": grid.best_params_,
        "best_cv_score": round(float(grid.best_score_), 4),
    }


@lru_cache(maxsize=64)
def _bagging_artifacts(
    dataset_key: str,
    n_estimators: int,
    max_samples: float,
    max_features: float,
    random_state: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    model = build_bagging(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
    )
    model.fit(bundle.X_train, bundle.y_train)
    metrics = classification_metrics(model, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test, bundle.target_names)
    metrics["oob_score"] = round(float(model.oob_score_), 4)
    metrics["oob_error"] = round(float(1 - model.oob_score_), 4)
    return {"model": model, "bundle": bundle, "metrics": metrics}


@lru_cache(maxsize=16)
def _tuned_bagging_artifacts(dataset_key: str, random_state: int) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    grid = tune_bagging(bundle.X_train, bundle.y_train, random_state=random_state)
    model = grid.best_estimator_
    metrics = classification_metrics(model, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test, bundle.target_names)
    metrics["oob_score"] = round(float(model.oob_score_), 4)
    metrics["oob_error"] = round(float(1 - model.oob_score_), 4)
    return {
        "model": model,
        "bundle": bundle,
        "metrics": metrics,
        "best_params": grid.best_params_,
        "best_cv_score": round(float(grid.best_score_), 4),
    }


@lru_cache(maxsize=64)
def _random_forest_artifacts(
    dataset_key: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str,
    random_state: int,
    oob_enabled: bool,
) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    model = build_random_forest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        oob_score=oob_enabled,
    )
    model.fit(bundle.X_train, bundle.y_train)
    metrics = classification_metrics(model, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test, bundle.target_names)
    if oob_enabled:
        metrics["oob_score"] = round(float(model.oob_score_), 4)
        metrics["oob_error"] = round(float(1 - model.oob_score_), 4)
    return {"model": model, "bundle": bundle, "metrics": metrics}


@lru_cache(maxsize=16)
def _tuned_random_forest_artifacts(dataset_key: str, random_state: int) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    grid = tune_random_forest(bundle.X_train, bundle.y_train, random_state=random_state)
    model = grid.best_estimator_
    metrics = classification_metrics(model, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test, bundle.target_names)
    metrics["oob_score"] = round(float(model.oob_score_), 4)
    metrics["oob_error"] = round(float(1 - model.oob_score_), 4)
    return {
        "model": model,
        "bundle": bundle,
        "metrics": metrics,
        "best_params": grid.best_params_,
        "best_cv_score": round(float(grid.best_score_), 4),
    }


@lru_cache(maxsize=64)
def _bagging_accuracy_curve(
    dataset_key: str,
    n_estimators: int,
    max_samples: float,
    max_features: float,
    random_state: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> tuple[list[int], list[float]]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    points = _curve_points(n_estimators)
    scores = []
    for point in points:
        model = build_bagging(
            n_estimators=point,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            oob_score=True,
        )
        model.fit(bundle.X_train, bundle.y_train)
        scores.append(round(float(model.score(bundle.X_test, bundle.y_test)), 4))
    return points, scores


@lru_cache(maxsize=64)
def _random_forest_accuracy_curve(
    dataset_key: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str,
    random_state: int,
) -> tuple[list[int], list[float]]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    points = _curve_points(n_estimators)
    scores = []
    for point in points:
        model = build_random_forest(
            n_estimators=point,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            oob_score=True,
        )
        model.fit(bundle.X_train, bundle.y_train)
        scores.append(round(float(model.score(bundle.X_test, bundle.y_test)), 4))
    return points, scores


def train_decision_tree_payload(dataset_key: str, max_depth: int | None, min_samples_leaf: int, random_state: int) -> dict[str, Any]:
    _validate_int("min_samples_leaf", min_samples_leaf, 1)
    if max_depth is not None:
        _validate_int("max_depth", max_depth, 1)
    _validate_int("random_state", random_state, 0)

    artifacts = _decision_tree_artifacts(dataset_key, max_depth, min_samples_leaf, random_state)
    model = artifacts["model"]
    bundle = artifacts["bundle"]
    metrics = artifacts["metrics"]
    return {
        "parameters": {
            "dataset_key": dataset_key,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
        },
        "metrics": metrics,
        "charts": {
            "confusion_matrix": _confusion_matrix_figure(metrics["confusion_matrix"], bundle.target_names, "Decision Tree Confusion Matrix"),
            "feature_importance": _feature_importance_figure(bundle.feature_names, model.feature_importances_.tolist(), "Decision Tree Feature Importance"),
            "metric_snapshot": _metric_snapshot_figure(metrics, "Decision Tree Performance Snapshot"),
            "prediction_distribution": _prediction_distribution_figure(metrics["prediction_distribution"], bundle.target_names, "Decision Tree Prediction Distribution"),
        },
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "feature_defaults": _feature_defaults(bundle),
        "sample_count": int(len(bundle.X_test)),
        "tree_rules": export_rules(model, bundle.feature_names, max_depth=4),
    }


def tune_decision_tree_payload(dataset_key: str, random_state: int) -> dict[str, Any]:
    _validate_int("random_state", random_state, 0)
    artifacts = _tuned_decision_tree_artifacts(dataset_key, random_state)
    payload = train_decision_tree_payload(
        dataset_key=dataset_key,
        max_depth=artifacts["best_params"].get("max_depth"),
        min_samples_leaf=int(artifacts["best_params"]["min_samples_leaf"]),
        random_state=random_state,
    )
    payload["best_params"] = artifacts["best_params"]
    payload["best_cv_score"] = artifacts["best_cv_score"]
    return payload


def train_bagging_payload(
    dataset_key: str,
    n_estimators: int,
    max_samples: float,
    max_features: float,
    random_state: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> dict[str, Any]:
    _validate_int("n_estimators", n_estimators, 1, 500)
    _validate_float("max_samples", max_samples, 0.1, 1.0)
    _validate_float("max_features", max_features, 0.1, 1.0)
    _validate_int("min_samples_leaf", min_samples_leaf, 1)
    if max_depth is not None:
        _validate_int("max_depth", max_depth, 1)
    _validate_int("random_state", random_state, 0)

    artifacts = _bagging_artifacts(dataset_key, n_estimators, max_samples, max_features, random_state, max_depth, min_samples_leaf)
    bundle = artifacts["bundle"]
    metrics = artifacts["metrics"]
    curve_x, curve_y = _bagging_accuracy_curve(dataset_key, n_estimators, max_samples, max_features, random_state, max_depth, min_samples_leaf)
    return {
        "parameters": {
            "dataset_key": dataset_key,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "random_state": random_state,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        },
        "metrics": metrics,
        "charts": {
            "confusion_matrix": _confusion_matrix_figure(metrics["confusion_matrix"], bundle.target_names, "Bagging Confusion Matrix"),
            "metric_snapshot": _metric_snapshot_figure(metrics, "Bagging Performance Snapshot"),
            "prediction_distribution": _prediction_distribution_figure(metrics["prediction_distribution"], bundle.target_names, "Bagging Prediction Distribution"),
            "accuracy_curve": _accuracy_curve_figure(curve_x, curve_y, "Bagging Accuracy vs Number of Estimators", "Estimators"),
        },
    }


def tune_bagging_payload(dataset_key: str, random_state: int) -> dict[str, Any]:
    _validate_int("random_state", random_state, 0)
    artifacts = _tuned_bagging_artifacts(dataset_key, random_state)
    params = artifacts["best_params"]
    payload = train_bagging_payload(
        dataset_key=dataset_key,
        n_estimators=int(params["n_estimators"]),
        max_samples=float(params["max_samples"]),
        max_features=float(params["max_features"]),
        random_state=random_state,
        max_depth=None,
        min_samples_leaf=1,
    )
    payload["best_params"] = artifacts["best_params"]
    payload["best_cv_score"] = artifacts["best_cv_score"]
    return payload


def train_random_forest_payload(
    dataset_key: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str,
    random_state: int,
    oob_enabled: bool,
) -> dict[str, Any]:
    _validate_int("n_estimators", n_estimators, 1, 500)
    _validate_int("min_samples_split", min_samples_split, 2)
    _validate_int("min_samples_leaf", min_samples_leaf, 1)
    if max_depth is not None:
        _validate_int("max_depth", max_depth, 1)
    _validate_int("random_state", random_state, 0)
    if max_features not in {"sqrt", "log2"}:
        raise ValueError("max_features must be either 'sqrt' or 'log2'.")

    artifacts = _random_forest_artifacts(
        dataset_key,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        random_state,
        oob_enabled,
    )
    model = artifacts["model"]
    bundle = artifacts["bundle"]
    metrics = artifacts["metrics"]
    curve_x, curve_y = _random_forest_accuracy_curve(
        dataset_key,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        random_state,
    )
    return {
        "parameters": {
            "dataset_key": dataset_key,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "oob_enabled": oob_enabled,
        },
        "metrics": metrics,
        "charts": {
            "confusion_matrix": _confusion_matrix_figure(metrics["confusion_matrix"], bundle.target_names, "Random Forest Confusion Matrix"),
            "metric_snapshot": _metric_snapshot_figure(metrics, "Random Forest Performance Snapshot"),
            "prediction_distribution": _prediction_distribution_figure(metrics["prediction_distribution"], bundle.target_names, "Random Forest Prediction Distribution"),
            "accuracy_curve": _accuracy_curve_figure(curve_x, curve_y, "Random Forest Accuracy vs Number of Trees", "Trees"),
            "feature_importance": _feature_importance_figure(bundle.feature_names, model.feature_importances_.tolist(), "Random Forest MDI Feature Importance"),
        },
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "feature_defaults": _feature_defaults(bundle),
        "sample_count": int(len(bundle.X_test)),
    }


def tune_random_forest_payload(dataset_key: str, random_state: int) -> dict[str, Any]:
    _validate_int("random_state", random_state, 0)
    artifacts = _tuned_random_forest_artifacts(dataset_key, random_state)
    params = artifacts["best_params"]
    payload = train_random_forest_payload(
        dataset_key=dataset_key,
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_split=2,
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=str(params["max_features"]),
        random_state=random_state,
        oob_enabled=True,
    )
    payload["best_params"] = artifacts["best_params"]
    payload["best_cv_score"] = artifacts["best_cv_score"]
    return payload


def _comparison_source_payloads(dataset_key: str, random_state: int, use_tuned_models: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if use_tuned_models:
        return (
            tune_decision_tree_payload(dataset_key, random_state),
            tune_bagging_payload(dataset_key, random_state),
            tune_random_forest_payload(dataset_key, random_state),
        )

    decision_tree_artifacts = _decision_tree_artifacts(dataset_key, 4, 1, random_state)
    bagging_artifacts = _bagging_artifacts(dataset_key, 50, 1.0, 1.0, random_state, 4, 1)
    random_forest_artifacts = _random_forest_artifacts(dataset_key, 100, 5, 2, 1, "sqrt", random_state, True)

    decision_tree = {
        "parameters": {"dataset_key": dataset_key, "max_depth": 4, "min_samples_leaf": 1, "random_state": random_state},
        "metrics": decision_tree_artifacts["metrics"],
    }
    bagging = {
        "parameters": {"dataset_key": dataset_key, "n_estimators": 50, "max_samples": 1.0, "max_features": 1.0, "random_state": random_state},
        "metrics": bagging_artifacts["metrics"],
    }
    random_forest = {
        "parameters": {"dataset_key": dataset_key, "n_estimators": 100, "max_depth": 5, "max_features": "sqrt", "random_state": random_state},
        "metrics": random_forest_artifacts["metrics"],
    }
    return decision_tree, bagging, random_forest


@lru_cache(maxsize=16)
def comparison_payload(dataset_key: str = "iris", random_state: int = 42, use_tuned_models: bool = True) -> dict[str, Any]:
    _validate_int("random_state", random_state, 0)
    decision_tree, bagging, random_forest = _comparison_source_payloads(dataset_key, random_state, use_tuned_models)

    models = ["Decision Tree", "Bagging", "Random Forest"]
    accuracy = [decision_tree["metrics"]["test_accuracy"], bagging["metrics"]["test_accuracy"], random_forest["metrics"]["test_accuracy"]]
    precision = [decision_tree["metrics"]["precision"], bagging["metrics"]["precision"], random_forest["metrics"]["precision"]]
    recall = [decision_tree["metrics"]["recall"], bagging["metrics"]["recall"], random_forest["metrics"]["recall"]]
    f1_score = [decision_tree["metrics"]["f1_score"], bagging["metrics"]["f1_score"], random_forest["metrics"]["f1_score"]]

    figure = go.Figure()
    figure.add_trace(go.Bar(name="Accuracy", x=models, y=accuracy, marker_color="#2563eb"))
    figure.add_trace(go.Bar(name="Precision", x=models, y=precision, marker_color="#14b8a6"))
    figure.add_trace(go.Bar(name="Recall", x=models, y=recall, marker_color="#f59e0b"))
    figure.add_trace(go.Bar(name="F1", x=models, y=f1_score, marker_color="#f97316"))
    figure.update_layout(
        barmode="group",
        title="Model Comparison on Shared Iris Split",
        margin=dict(l=40, r=20, t=55, b=40),
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    rows = []
    for model_name, payload in zip(models, [decision_tree, bagging, random_forest]):
        metrics = payload["metrics"]
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["test_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "cv_accuracy": metrics["cv_accuracy"],
                "train_accuracy": metrics["train_accuracy"],
            }
        )

    summary = dataset_summary(dataset_key)
    best_row = max(rows, key=lambda row: row["accuracy"])
    summary["best_model_accuracy"] = best_row["accuracy"]
    if best_row["model"] == "Bagging":
        summary["number_of_estimators"] = int(bagging["parameters"]["n_estimators"])
    elif best_row["model"] == "Random Forest":
        summary["number_of_estimators"] = int(random_forest["parameters"]["n_estimators"])
    else:
        summary["number_of_estimators"] = 1

    return {
        "summary": summary,
        "rows": rows,
        "charts": {"comparison": figure_to_dict(figure)},
        "decision_tree": decision_tree,
        "bagging": bagging,
        "random_forest": random_forest,
    }


def tuned_random_forest_artifacts(dataset_key: str = "iris", random_state: int = 42) -> dict[str, Any]:
    """Expose cached tuned random forest artifacts to other services."""
    return _tuned_random_forest_artifacts(dataset_key, random_state)


def decision_tree_explorer_payload(
    dataset_key: str,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
    sample_index: int,
    feature_x: str,
    feature_y: str,
) -> dict[str, Any]:
    artifacts = _decision_tree_artifacts(dataset_key, max_depth, min_samples_leaf, random_state)
    model = artifacts["model"]
    bundle = artifacts["bundle"]
    _validate_int("sample_index", sample_index, 0, len(bundle.X_test) - 1)
    if feature_x not in bundle.feature_names or feature_y not in bundle.feature_names or feature_x == feature_y:
        raise ValueError("Choose two different valid features for the decision boundary explorer.")

    sample_row = bundle.X_test.iloc[sample_index]
    true_label = bundle.target_names[int(bundle.y_test.iloc[sample_index])]
    predicted_index = int(model.predict(sample_row.to_frame().T)[0])
    predicted_label = bundle.target_names[predicted_index]
    probabilities = model.predict_proba(sample_row.to_frame().T)[0].tolist()
    return {
        "sample_index": sample_index,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "sample_features": {feature_name: round(float(sample_row[feature_name]), 3) for feature_name in bundle.feature_names},
        "path_rows": _decision_path_details(model, sample_row, bundle.feature_names, bundle.target_names),
        "charts": {
            "probabilities": _probability_figure(bundle.target_names, probabilities, "Decision Tree Class Probabilities"),
            "decision_boundary": _decision_boundary_figure(model, bundle, feature_x, feature_y, "Decision Tree Decision Boundary Explorer"),
        },
    }


def random_forest_voting_payload(
    dataset_key: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str,
    random_state: int,
    sample_index: int,
) -> dict[str, Any]:
    artifacts = _random_forest_artifacts(
        dataset_key,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        random_state,
        True,
    )
    model = artifacts["model"]
    bundle = artifacts["bundle"]
    _validate_int("sample_index", sample_index, 0, len(bundle.X_test) - 1)
    sample_row = bundle.X_test.iloc[sample_index]
    votes = [int(estimator.predict(sample_row.to_frame().T)[0]) for estimator in model.estimators_]
    vote_counts = np.bincount(votes, minlength=len(bundle.target_names)).tolist()
    vote_count_figure = go.Figure(
        go.Bar(
            x=bundle.target_names,
            y=vote_counts,
            marker=dict(color=["#2563eb", "#0f766e", "#f97316"][: len(bundle.target_names)]),
        )
    )
    vote_count_figure.update_layout(
        title="Random Forest Vote Count by Class",
        margin=dict(l=40, r=20, t=55, b=40),
        yaxis_title="Votes",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    tree_index = list(range(1, len(votes[:40]) + 1))
    vote_trace_figure = go.Figure(
        go.Scatter(
            x=tree_index,
            y=votes[:40],
            mode="markers+lines",
            marker=dict(size=9, color=votes[:40], colorscale="Viridis"),
            line=dict(color="#64748b", width=1),
            hovertemplate="Tree %{x}<br>Predicted class index %{y}<extra></extra>",
        )
    )
    vote_trace_figure.update_layout(
        title="First Trees and Their Votes",
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Tree number",
        yaxis_title="Predicted class index",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    probabilities = model.predict_proba(sample_row.to_frame().T)[0].tolist()
    predicted_label = bundle.target_names[int(model.predict(sample_row.to_frame().T)[0])]
    true_label = bundle.target_names[int(bundle.y_test.iloc[sample_index])]
    return {
        "sample_index": sample_index,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "vote_counts": [
            {"label": bundle.target_names[index], "votes": vote_counts[index]}
            for index in range(len(bundle.target_names))
        ],
        "sample_features": {feature_name: round(float(sample_row[feature_name]), 3) for feature_name in bundle.feature_names},
        "charts": {
            "vote_counts": figure_to_dict(vote_count_figure),
            "tree_votes": figure_to_dict(vote_trace_figure),
            "probabilities": _probability_figure(bundle.target_names, probabilities, "Random Forest Consensus Probabilities"),
        },
    }


def prediction_playground_payload(model_type: str, features: dict[str, float], dataset_key: str = "iris", random_state: int = 42) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key, random_state=DEFAULT_SPLIT_RANDOM_STATE)
    missing = [feature_name for feature_name in bundle.feature_names if feature_name not in features]
    if missing:
        raise ValueError(f"Missing feature values: {', '.join(missing)}")

    import pandas as pd

    sample = pd.DataFrame(
        [[float(features[feature_name]) for feature_name in bundle.feature_names]],
        columns=bundle.feature_names,
    )
    if model_type == "decision_tree":
        artifacts = _tuned_decision_tree_artifacts(dataset_key, random_state)
    elif model_type == "bagging":
        artifacts = _tuned_bagging_artifacts(dataset_key, random_state)
    elif model_type == "random_forest":
        artifacts = _tuned_random_forest_artifacts(dataset_key, random_state)
    else:
        raise ValueError("model_type must be decision_tree, bagging, or random_forest.")

    model = artifacts["model"]
    probabilities = model.predict_proba(sample)[0].tolist()
    predicted_index = int(model.predict(sample)[0])
    explanation_rows = _feature_explanation_rows(bundle.feature_names, getattr(model, "feature_importances_", np.zeros(len(bundle.feature_names))).tolist())
    return {
        "model_type": model_type,
        "predicted_label": bundle.target_names[predicted_index],
        "probabilities": [
            {"label": bundle.target_names[index], "probability": round(float(probabilities[index]), 4)}
            for index in range(len(bundle.target_names))
        ],
        "top_features": explanation_rows,
        "charts": {
            "probabilities": _probability_figure(bundle.target_names, probabilities, "Prediction Playground Probabilities"),
        },
    }
