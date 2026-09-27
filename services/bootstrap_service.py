from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from models.data_loader import load_dataset
from services.model_service import figure_to_dict


def _validate_inputs(sample_size: int, n_samples: int, random_seed: int, dataset_size: int) -> None:
    if sample_size < 1 or sample_size > dataset_size:
        raise ValueError(f"sample_size must be between 1 and {dataset_size}.")
    if n_samples < 1 or n_samples > 12:
        raise ValueError("number of bootstrap samples must be between 1 and 12.")
    if random_seed < 0:
        raise ValueError("random seed must be non-negative.")


def generate_bootstrap_payload(dataset_key: str, sample_size: int, n_samples: int, random_seed: int) -> dict[str, Any]:
    dataset_key, dataset_label, X, y, feature_names, target_names = load_dataset(dataset_key)
    dataset_size = len(X)
    _validate_inputs(sample_size, n_samples, random_seed, dataset_size)

    rng = np.random.default_rng(random_seed)
    all_indices = np.arange(dataset_size)
    frequency_matrix = []
    sample_summaries = []
    aggregate_counts = np.zeros(dataset_size, dtype=int)

    for sample_number in range(1, n_samples + 1):
        sampled_indices = rng.choice(all_indices, size=sample_size, replace=True)
        counts = np.bincount(sampled_indices, minlength=dataset_size)
        aggregate_counts += counts
        unique_indices = np.flatnonzero(counts > 0)
        repeated_indices = np.flatnonzero(counts > 1)
        oob_indices = np.flatnonzero(counts == 0)
        frequency_matrix.append(counts.tolist())
        sample_summaries.append(
            {
                "sample_number": sample_number,
                "indices": sampled_indices.astype(int).tolist(),
                "counts": counts.astype(int).tolist(),
                "unique_count": int(len(unique_indices)),
                "oob_count": int(len(oob_indices)),
                "included_pct": round(float(len(unique_indices) / sample_size * 100), 2),
                "excluded_pct": round(float(len(oob_indices) / dataset_size * 100), 2),
                "repeated_indices": repeated_indices.astype(int).tolist(),
                "oob_indices": oob_indices.astype(int).tolist(),
            }
        )

    original_rows = []
    target_lookup = {index: name for index, name in enumerate(target_names)}
    combined = X.copy()
    combined["species"] = y.map(target_lookup)
    for row_id, row in combined.reset_index(drop=True).iterrows():
        original_rows.append(
            {
                "row_id": int(row_id),
                "features": {feature_name: round(float(row[feature_name]), 2) for feature_name in feature_names},
                "species": str(row["species"]),
            }
        )

    mean_unique = round(float(np.mean([item["unique_count"] for item in sample_summaries])), 2)
    mean_oob = round(float(np.mean([item["oob_count"] for item in sample_summaries])), 2)

    frequency_figure = go.Figure(
        go.Bar(
            x=list(range(dataset_size)),
            y=aggregate_counts.tolist(),
            marker=dict(color=aggregate_counts.tolist(), colorscale="Blues"),
            hovertemplate="Observation %{x}<br>Frequency %{y}<extra></extra>",
        )
    )
    frequency_figure.update_layout(
        title="Observation Frequency Across Bootstrap Samples",
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Observation index",
        yaxis_title="Total sampled count",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    heatmap_figure = go.Figure(
        go.Heatmap(
            z=frequency_matrix,
            x=list(range(dataset_size)),
            y=[f"Sample {item['sample_number']}" for item in sample_summaries],
            colorscale="Teal",
            hovertemplate="%{y}<br>Observation %{x}<br>Count %{z}<extra></extra>",
        )
    )
    heatmap_figure.update_layout(
        title="Repeated Rows and OOB Gaps by Sample",
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Observation index",
        yaxis_title="Bootstrap sample",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    donut_figure = go.Figure(
        go.Pie(
            labels=["Average unique observations", "Average OOB observations"],
            values=[mean_unique, mean_oob],
            hole=0.58,
            marker=dict(colors=["#0f766e", "#f97316"]),
        )
    )
    donut_figure.update_layout(
        title="Average Included vs Excluded Observations",
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return {
        "parameters": {
            "dataset_key": dataset_key,
            "sample_size": sample_size,
            "n_samples": n_samples,
            "random_seed": random_seed,
        },
        "dataset_label": dataset_label,
        "feature_names": feature_names,
        "dataset_size": dataset_size,
        "original_rows": original_rows,
        "sample_summaries": sample_summaries,
        "mean_unique": mean_unique,
        "mean_oob": mean_oob,
        "charts": {
            "frequency": figure_to_dict(frequency_figure),
            "heatmap": figure_to_dict(heatmap_figure),
            "composition": figure_to_dict(donut_figure),
        },
    }
