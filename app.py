from __future__ import annotations

import os

import plotly.graph_objects as go
from flask import Flask, jsonify, render_template, request, url_for

from models.data_loader import available_datasets, dataset_columns, dataset_profile, save_uploaded_dataset, update_dataset_target_column

from services.bootstrap_service import generate_bootstrap_payload
from services.dataset_profile_service import dataset_profile_payload
from services.feature_importance import feature_importance_payload
from services.model_service import (
    comparison_payload,
    decision_tree_explorer_payload,
    figure_to_dict,
    prediction_playground_payload,
    random_forest_voting_payload,
    train_bagging_payload,
    train_decision_tree_payload,
    train_random_forest_payload,
    tune_bagging_payload,
    tune_decision_tree_payload,
    tune_random_forest_payload,
)
from services.oob_service import oob_payload
from services.run_history_service import load_run_history, save_run_history_entry, update_run_history_notes

app = Flask(__name__)

NAV_SECTIONS = [
    {"key": "home", "label": "Home", "endpoint": "home_page"},
    {"key": "overview", "label": "Overview", "endpoint": "overview_page"},
    {"key": "dataset_profile", "label": "Dataset Profile", "endpoint": "dataset_profile_page"},
    {"key": "bootstrap", "label": "Bootstrap Sampling", "endpoint": "bootstrap_page"},
    {"key": "decision_tree", "label": "Decision Tree", "endpoint": "decision_tree_page"},
    {"key": "bagging", "label": "Bagging", "endpoint": "bagging_page"},
    {"key": "random_forest", "label": "Random Forest", "endpoint": "random_forest_page"},
    {"key": "comparison", "label": "Model Comparison", "endpoint": "comparison_page"},
    {"key": "importance", "label": "Feature Importance", "endpoint": "importance_page"},
    {"key": "oob", "label": "OOB Analysis", "endpoint": "oob_page"},
    {"key": "bias_variance", "label": "Bias vs Variance", "endpoint": "bias_variance_page"},
    {"key": "experiments", "label": "Experiments", "endpoint": "experiments_page"},
    {"key": "api_reference", "label": "API Reference", "endpoint": "api_reference_page"},
]


@app.context_processor
def inject_navigation() -> dict[str, list[dict[str, str]]]:
    current_dataset_key = request.args.get("dataset", "iris")
    return {
        "nav_sections": [
            {
                "key": item["key"],
                "label": item["label"],
                "url": url_for(item["endpoint"], dataset=current_dataset_key),
            }
            for item in NAV_SECTIONS
        ],
        "dataset_options": available_datasets(),
        "current_dataset_key": current_dataset_key,
    }


def _render_page(template_name: str, page_key: str, title: str, subtitle: str, initial_data: dict, dataset_key: str):
    return render_template(
        template_name,
        page_key=page_key,
        page_title=title,
        page_subtitle=subtitle,
        initial_data=initial_data,
        current_route_path=request.path,
        current_dataset_key=dataset_key,
    )


def _home_payload() -> dict:
    dataset_cards = []
    for option in available_datasets():
        profile = dataset_profile(option["key"])
        dataset_cards.append(
            {
                "key": option["key"],
                "label": option["label"],
                "source_type": option.get("source_type", "builtin"),
                "samples": profile.row_count,
                "features": profile.column_count - 1,
                "target_column": profile.target_column,
                "task_type": profile.task_type,
            }
        )
    return {"datasets": dataset_cards}


def _payload() -> dict:
    return request.get_json(silent=True) or {}


def _parse_optional_int(value):
    if value in (None, "", "None"):
        return None
    return int(value)


def _json_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _dataset_key(default: str = "iris") -> str:
    if request.method == "GET":
        return request.args.get("dataset", default)
    return str(_payload().get("dataset_key", default))


def _build_bias_variance_data(dataset_key: str) -> dict:
    comparison = comparison_payload(dataset_key, 42, use_tuned_models=False)
    rows = comparison["rows"]
    models = [row["model"] for row in rows]

    conceptual_figure = go.Figure()
    conceptual_figure.add_trace(
        go.Bar(
            name="Relative bias level",
            x=models,
            y=[2.0, 2.1, 2.4],
            marker_color="#f59e0b",
        )
    )
    conceptual_figure.add_trace(
        go.Bar(
            name="Relative variance level",
            x=models,
            y=[3.0, 1.9, 1.3],
            marker_color="#0f766e",
        )
    )
    conceptual_figure.update_layout(
        barmode="group",
        title="Conceptual Bias-Variance Tradeoff (Relative Scale)",
        margin=dict(l=40, r=20, t=55, b=40),
        yaxis_title="Relative conceptual scale",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    gap_figure = go.Figure()
    gap_figure.add_trace(
        go.Bar(
            name="Train accuracy",
            x=models,
            y=[row["train_accuracy"] for row in rows],
            marker_color="#2563eb",
        )
    )
    gap_figure.add_trace(
        go.Bar(
            name="Test accuracy",
            x=models,
            y=[row["accuracy"] for row in rows],
            marker_color="#f97316",
        )
    )
    gap_figure.update_layout(
        barmode="group",
        title="Observed Train vs Test Accuracy Gap",
        margin=dict(l=40, r=20, t=55, b=40),
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return {
        "charts": {
            "bias_variance": figure_to_dict(conceptual_figure),
            "generalization_gap": figure_to_dict(gap_figure),
        }
    }


@app.route("/")
def home_page():
    return render_template(
        "home.html",
        page_title="ML Ensemble Lab",
        initial_data=_home_payload(),
        dataset_options=available_datasets(),
    )


@app.route("/overview")
def overview_page():
    dataset_key = _dataset_key()
    payload = comparison_payload(dataset_key, 42, use_tuned_models=False)
    profile = dataset_profile(dataset_key)
    profile_payload = dataset_profile_payload(dataset_key)
    payload["profile"] = {
        "target_column": profile.target_column,
        "task_type": profile.task_type,
        "missing_values": profile.missing_values,
        "duplicate_rows": profile.duplicate_rows,
        "warnings": profile.warnings,
        "class_balance": profile.class_balance,
        "dtype_summary": profile.dtype_summary,
    }
    payload["profile_charts"] = profile_payload["charts"]
    payload["run_history"] = load_run_history(dataset_key, limit=12)
    return _render_page(
        "overview.html",
        "overview",
        "Ensemble Learning",
        "A professional, interactive dashboard built around the actual notebook implementation.",
        payload,
        dataset_key,
    )


@app.route("/dataset-profile")
def dataset_profile_page():
    dataset_key = _dataset_key()
    selected_column = request.args.get("column")
    payload = dataset_profile_payload(dataset_key, selected_column)
    payload["target_columns"] = dataset_columns(dataset_key)
    payload["is_target_editable"] = dataset_key != "iris"
    return _render_page(
        "dataset_profile.html",
        "dataset_profile",
        "Dataset Profile",
        "Inspect correlations, target distribution, missingness, and per-column behavior before running ensemble experiments.",
        payload,
        dataset_key,
    )


@app.route("/bootstrap")
def bootstrap_page():
    dataset_key = _dataset_key()
    return _render_page(
        "bootstrap.html",
        "bootstrap",
        "Bootstrap Sampling",
        "See sampling with replacement, repeated rows, and OOB observations unfold visually.",
        generate_bootstrap_payload(dataset_key=dataset_key, sample_size=150, n_samples=4, random_seed=42),
        dataset_key,
    )


@app.route("/decision-tree")
def decision_tree_page():
    dataset_key = _dataset_key()
    return _render_page(
        "decision_tree.html",
        "decision_tree",
        "Decision Tree",
        "Interact with the base learner used throughout the bagging and random forest workflow.",
        train_decision_tree_payload(dataset_key=dataset_key, max_depth=4, min_samples_leaf=1, random_state=42),
        dataset_key,
    )


@app.route("/bagging")
def bagging_page():
    dataset_key = _dataset_key()
    return _render_page(
        "bagging.html",
        "bagging",
        "Bagging",
        "Notebook-aligned bootstrap aggregation with live metrics and estimator curves.",
        train_bagging_payload(
            dataset_key=dataset_key,
            n_estimators=50,
            max_samples=1.0,
            max_features=1.0,
            random_state=42,
            max_depth=4,
            min_samples_leaf=1,
        ),
        dataset_key,
    )


@app.route("/random-forest")
def random_forest_page():
    dataset_key = _dataset_key()
    return _render_page(
        "random_forest.html",
        "random_forest",
        "Random Forest",
        "Bootstrap sampling plus random feature selection with live OOB-aware metrics.",
        train_random_forest_payload(
            dataset_key=dataset_key,
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            oob_enabled=True,
        ),
        dataset_key,
    )


@app.route("/comparison")
def comparison_page():
    dataset_key = _dataset_key()
    return _render_page(
        "comparison.html",
        "comparison",
        "Model Comparison",
        "Decision Tree, Bagging, and Random Forest compared on the same held-out split.",
        comparison_payload(dataset_key, 42, use_tuned_models=False),
        dataset_key,
    )


@app.route("/feature-importance")
def importance_page():
    dataset_key = _dataset_key()
    return _render_page(
        "importance.html",
        "importance",
        "Feature Importance",
        "Switch between MDI / Gini importance and permutation importance on live model results.",
        feature_importance_payload(dataset_key=dataset_key, random_state=42, n_repeats=10, include_permutation=False, use_tuned_model=False),
        dataset_key,
    )


@app.route("/oob")
def oob_page():
    dataset_key = _dataset_key()
    return _render_page(
        "oob.html",
        "oob",
        "Out-of-Bag Analysis",
        "Use actual random forest results to inspect how OOB score changes with the number of trees.",
        oob_payload(dataset_key=dataset_key, random_state=42, max_estimators=100, use_tuned_model=False),
        dataset_key,
    )


@app.route("/bias-variance")
def bias_variance_page():
    dataset_key = _dataset_key()
    return _render_page(
        "bias_variance.html",
        "bias_variance",
        "Bias vs Variance",
        "A notebook-faithful conceptual view backed by actual train/test behavior from the tuned models.",
        _build_bias_variance_data(dataset_key),
        dataset_key,
    )


@app.route("/api-reference")
def api_reference_page():
    dataset_key = _dataset_key()
    api_groups = [
        {
            "title": "Page Routes",
            "items": [
                {"method": "GET", "path": "/", "description": "Overview dashboard"},
                {"method": "GET", "path": "/bootstrap", "description": "Bootstrap sampling page"},
                {"method": "GET", "path": "/decision-tree", "description": "Decision Tree explorer"},
                {"method": "GET", "path": "/bagging", "description": "Bagging dashboard"},
                {"method": "GET", "path": "/random-forest", "description": "Random Forest dashboard"},
                {"method": "GET", "path": "/comparison", "description": "Model comparison page"},
                {"method": "GET", "path": "/feature-importance", "description": "Importance analysis page"},
                {"method": "GET", "path": "/oob", "description": "OOB analysis page"},
                {"method": "GET", "path": "/bias-variance", "description": "Bias vs variance page"},
            ],
        },
        {
            "title": "Interactive APIs",
            "items": [
                {
                    "method": "POST",
                    "path": "/api/bootstrap",
                    "description": "Generate bootstrap samples",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "sample_size": 150,
                        "n_samples": 4,
                        "random_seed": 42,
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/decision-tree/train",
                    "description": "Train Decision Tree",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "max_depth": 4,
                        "min_samples_leaf": 1,
                        "random_state": 42,
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/decision-tree/tune",
                    "description": "Tune Decision Tree",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "random_state": 42,
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/decision-tree/explorer",
                    "description": "Inspect sample path and boundary",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "max_depth": 4,
                        "min_samples_leaf": 1,
                        "random_state": 42,
                        "sample_index": 0,
                        "feature_x": "petal length (cm)" if dataset_key == "iris" else "BP",
                        "feature_y": "petal width (cm)" if dataset_key == "iris" else "cholestrol",
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/bagging/train",
                    "description": "Train Bagging model",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "n_estimators": 50,
                        "max_samples": 1.0,
                        "max_features": 1.0,
                        "random_state": 42,
                        "max_depth": 4,
                        "min_samples_leaf": 1,
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/bagging/tune",
                    "description": "Tune Bagging model",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "random_state": 42,
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/random-forest/train",
                    "description": "Train Random Forest",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "n_estimators": 100,
                        "max_depth": 5,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "random_state": 42,
                        "oob_enabled": True,
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/random-forest/tune",
                    "description": "Tune Random Forest",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "random_state": 42,
                    },
                },
                {
                    "method": "POST",
                    "path": "/api/random-forest/voting",
                    "description": "Inspect tree votes",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "n_estimators": 100,
                        "max_depth": 5,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "random_state": 42,
                        "sample_index": 0,
                    },
                },
                {
                    "method": "GET",
                    "path": "/api/model-comparison",
                    "description": "Get model comparison data",
                    "query": f"?dataset={dataset_key}&random_state=42",
                    "try_live": True,
                },
                {
                    "method": "GET",
                    "path": "/api/feature-importance",
                    "description": "Get feature-importance data",
                    "query": f"?dataset={dataset_key}&random_state=42&n_repeats=10",
                    "try_live": True,
                },
                {
                    "method": "GET",
                    "path": "/api/oob",
                    "description": "Get OOB curve data",
                    "query": f"?dataset={dataset_key}&random_state=42&max_estimators=200",
                    "try_live": True,
                },
                {
                    "method": "POST",
                    "path": "/api/predict",
                    "description": "Prediction playground",
                    "sample_body": {
                        "dataset_key": dataset_key,
                        "model_type": "random_forest",
                        "random_state": 42,
                        "features": {
                            "sepal length (cm)": 5.1,
                            "sepal width (cm)": 3.5,
                            "petal length (cm)": 1.4,
                            "petal width (cm)": 0.2,
                        }
                        if dataset_key == "iris"
                        else {
                            "age": 54,
                            "sex": 1,
                            "BP": 130,
                            "cholestrol": 250,
                        },
                    },
                },
            ],
        },
    ]
    return _render_page(
        "api_reference.html",
        "api_reference",
        "Reusable Flask APIs",
        "Live endpoint reference for the interactive dashboard, with dataset-aware routes and examples.",
        {"api_groups": api_groups},
        dataset_key,
    )


@app.route("/experiments")
def experiments_page():
    dataset_key = _dataset_key()
    return _render_page(
        "experiments.html",
        "experiments",
        "Experiments",
        "Persistent experiment records with side-by-side diffing and note editing for the current dataset.",
        {"rows": load_run_history(dataset_key, 50)},
        dataset_key,
    )


@app.route("/api/bootstrap", methods=["POST"])
def bootstrap_api():
    payload = _payload()
    try:
        return jsonify(
            generate_bootstrap_payload(
                dataset_key=str(payload.get("dataset_key", "iris")),
                sample_size=int(payload.get("sample_size", 150)),
                n_samples=int(payload.get("n_samples", 4)),
                random_seed=int(payload.get("random_seed", 42)),
            )
        )
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Bootstrap generation failed. Please try different inputs.", 500)


@app.route("/api/decision-tree/train", methods=["POST"])
def decision_tree_train_api():
    payload = _payload()
    try:
        return jsonify(
            train_decision_tree_payload(
                dataset_key=str(payload.get("dataset_key", "iris")),
                max_depth=_parse_optional_int(payload.get("max_depth")),
                min_samples_leaf=int(payload.get("min_samples_leaf", 1)),
                random_state=int(payload.get("random_state", 42)),
            )
        )
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Decision Tree training failed.", 500)


@app.route("/api/decision-tree/tune", methods=["POST"])
def decision_tree_tune_api():
    payload = _payload()
    try:
        return jsonify(tune_decision_tree_payload(str(payload.get("dataset_key", "iris")), random_state=int(payload.get("random_state", 42))))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Decision Tree hyperparameter tuning failed.", 500)


@app.route("/api/decision-tree/explorer", methods=["POST"])
def decision_tree_explorer_api():
    payload = _payload()
    try:
        return jsonify(
            decision_tree_explorer_payload(
                dataset_key=str(payload.get("dataset_key", "iris")),
                max_depth=_parse_optional_int(payload.get("max_depth")),
                min_samples_leaf=int(payload.get("min_samples_leaf", 1)),
                random_state=int(payload.get("random_state", 42)),
                sample_index=int(payload.get("sample_index", 0)),
                feature_x=str(payload.get("feature_x", "petal length (cm)")),
                feature_y=str(payload.get("feature_y", "petal width (cm)")),
            )
        )
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Decision Tree explorer failed.", 500)


@app.route("/api/bagging/train", methods=["POST"])
def bagging_train_api():
    payload = _payload()
    try:
        return jsonify(
            train_bagging_payload(
                dataset_key=str(payload.get("dataset_key", "iris")),
                n_estimators=int(payload.get("n_estimators", 50)),
                max_samples=float(payload.get("max_samples", 1.0)),
                max_features=float(payload.get("max_features", 1.0)),
                random_state=int(payload.get("random_state", 42)),
                max_depth=_parse_optional_int(payload.get("max_depth")),
                min_samples_leaf=int(payload.get("min_samples_leaf", 1)),
            )
        )
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Bagging model training failed.", 500)


@app.route("/api/bagging/tune", methods=["POST"])
def bagging_tune_api():
    payload = _payload()
    try:
        return jsonify(tune_bagging_payload(str(payload.get("dataset_key", "iris")), random_state=int(payload.get("random_state", 42))))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Bagging hyperparameter tuning failed.", 500)


@app.route("/api/random-forest/train", methods=["POST"])
def random_forest_train_api():
    payload = _payload()
    try:
        return jsonify(
            train_random_forest_payload(
                dataset_key=str(payload.get("dataset_key", "iris")),
                n_estimators=int(payload.get("n_estimators", 100)),
                max_depth=_parse_optional_int(payload.get("max_depth")),
                min_samples_split=int(payload.get("min_samples_split", 2)),
                min_samples_leaf=int(payload.get("min_samples_leaf", 1)),
                max_features=str(payload.get("max_features", "sqrt")),
                random_state=int(payload.get("random_state", 42)),
                oob_enabled=bool(payload.get("oob_enabled", True)),
            )
        )
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Random Forest training failed.", 500)


@app.route("/api/random-forest/tune", methods=["POST"])
def random_forest_tune_api():
    payload = _payload()
    try:
        return jsonify(tune_random_forest_payload(str(payload.get("dataset_key", "iris")), random_state=int(payload.get("random_state", 42))))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Random Forest hyperparameter tuning failed.", 500)


@app.route("/api/random-forest/voting", methods=["POST"])
def random_forest_voting_api():
    payload = _payload()
    try:
        return jsonify(
            random_forest_voting_payload(
                dataset_key=str(payload.get("dataset_key", "iris")),
                n_estimators=int(payload.get("n_estimators", 100)),
                max_depth=_parse_optional_int(payload.get("max_depth")),
                min_samples_split=int(payload.get("min_samples_split", 2)),
                min_samples_leaf=int(payload.get("min_samples_leaf", 1)),
                max_features=str(payload.get("max_features", "sqrt")),
                random_state=int(payload.get("random_state", 42)),
                sample_index=int(payload.get("sample_index", 0)),
            )
        )
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Random Forest voting explorer failed.", 500)


@app.route("/api/model-comparison")
def comparison_api():
    try:
        dataset_key = request.args.get("dataset", "iris")
        random_state = int(request.args.get("random_state", 42))
        return jsonify(comparison_payload(dataset_key, random_state))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Model comparison could not be generated.", 500)


@app.route("/api/feature-importance")
def feature_importance_api():
    try:
        dataset_key = request.args.get("dataset", "iris")
        random_state = int(request.args.get("random_state", 42))
        n_repeats = int(request.args.get("n_repeats", 10))
        include_permutation = request.args.get("include_permutation", "true").lower() != "false"
        return jsonify(feature_importance_payload(dataset_key=dataset_key, random_state=random_state, n_repeats=n_repeats, include_permutation=include_permutation))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Feature importance analysis failed.", 500)


@app.route("/api/oob")
def oob_api():
    try:
        dataset_key = request.args.get("dataset", "iris")
        random_state = int(request.args.get("random_state", 42))
        max_estimators = int(request.args.get("max_estimators", 200))
        return jsonify(oob_payload(dataset_key=dataset_key, random_state=random_state, max_estimators=max_estimators))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("OOB analysis failed.", 500)


@app.route("/api/predict", methods=["POST"])
def prediction_playground_api():
    payload = _payload()
    try:
        return jsonify(
            prediction_playground_payload(
                model_type=str(payload.get("model_type", "random_forest")),
                features=dict(payload.get("features", {})),
                dataset_key=str(payload.get("dataset_key", "iris")),
                random_state=int(payload.get("random_state", 42)),
            )
        )
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        return _json_error("Prediction playground failed.", 500)


@app.route("/api/run-history", methods=["GET", "POST"])
def run_history_api():
    if request.method == "GET":
        dataset_key = request.args.get("dataset")
        limit = int(request.args.get("limit", 25))
        return jsonify({"rows": load_run_history(dataset_key, limit)})

    payload = _payload()
    try:
        record = save_run_history_entry(payload)
        return jsonify(record)
    except Exception as exc:
        return _json_error(f"Failed to save run history: {exc}", 500)


@app.route("/api/run-history/<run_id>", methods=["PUT"])
def run_history_update_api(run_id: str):
    payload = _payload()
    try:
        return jsonify(update_run_history_notes(run_id, str(payload.get("notes", ""))))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception as exc:
        return _json_error(f"Failed to update run history: {exc}", 500)


@app.route("/api/datasets/upload", methods=["POST"])
def dataset_upload_api():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return _json_error("Please choose a CSV file to upload.")
    if not uploaded_file.filename.lower().endswith(".csv"):
        return _json_error("Only CSV files are supported right now.")

    try:
        dataset_info = save_uploaded_dataset(uploaded_file.filename, uploaded_file.read())
        profile = dataset_profile(dataset_info["dataset_key"])
        return jsonify(
            {
                "dataset_key": dataset_info["dataset_key"],
                "dataset_label": dataset_info["dataset_label"],
                "profile": {
                    "samples": profile.row_count,
                    "features": profile.column_count - 1,
                    "target_column": profile.target_column,
                    "task_type": profile.task_type,
                    "warnings": profile.warnings,
                },
                "redirect_url": url_for("overview_page", dataset=dataset_info["dataset_key"]),
            }
        )
    except Exception as exc:
        return _json_error(f"Dataset upload failed: {exc}", 500)


@app.route("/api/datasets/target-column", methods=["POST"])
def dataset_target_column_api():
    payload = _payload()
    try:
        result = update_dataset_target_column(str(payload.get("dataset_key", "")), str(payload.get("target_column", "")))
        return jsonify({**result, "redirect_url": url_for("dataset_profile_page", dataset=result["dataset_key"])})
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception as exc:
        return _json_error(f"Target-column update failed: {exc}", 500)


@app.route("/api/dataset-profile")
def dataset_profile_api():
    try:
        dataset_key = request.args.get("dataset", "iris")
        selected_column = request.args.get("column")
        return jsonify(dataset_profile_payload(dataset_key, selected_column))
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception as exc:
        return _json_error(f"Dataset profile failed: {exc}", 500)


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
