from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from models.data_loader import dataset_columns, dataset_profile, load_dataset_bundle
from services.model_service import figure_to_dict


def _column_profile(frame: pd.DataFrame, column_name: str) -> dict[str, Any]:
    series = frame[column_name]
    summary: dict[str, Any] = {
        "column": column_name,
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        summary.update(
            {
                "min": round(float(series.min()), 4) if not series.dropna().empty else None,
                "max": round(float(series.max()), 4) if not series.dropna().empty else None,
                "mean": round(float(series.mean()), 4) if not series.dropna().empty else None,
                "median": round(float(series.median()), 4) if not series.dropna().empty else None,
            }
        )
    else:
        top_values = series.astype(str).value_counts().head(8)
        summary["top_values"] = [{"label": index, "count": int(count)} for index, count in top_values.items()]
    return summary


def dataset_profile_payload(dataset_key: str, selected_column: str | None = None) -> dict[str, Any]:
    bundle = load_dataset_bundle(dataset_key=dataset_key)
    profile = dataset_profile(dataset_key)
    frame = bundle.original_frame.copy()
    available_columns = dataset_columns(dataset_key)
    active_column = selected_column if selected_column in available_columns else available_columns[0]

    target_series = frame[profile.target_column]
    if profile.task_type == "classification":
        target_distribution = target_series.astype("category").value_counts().sort_index()
        target_figure = go.Figure(
            go.Bar(
                x=[str(index) for index in target_distribution.index],
                y=target_distribution.values.tolist(),
                marker=dict(color="#2563eb"),
            )
        )
        target_figure.update_layout(
            title="Target Distribution",
            margin=dict(l=40, r=20, t=55, b=40),
            xaxis_title="Class",
            yaxis_title="Count",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
    else:
        target_figure = go.Figure(
            go.Histogram(x=target_series, marker=dict(color="#2563eb"))
        )
        target_figure.update_layout(
            title="Target Distribution",
            margin=dict(l=40, r=20, t=55, b=40),
            xaxis_title=profile.target_column,
            yaxis_title="Count",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    if profile.correlation_matrix and profile.correlation_columns:
        corr_figure = go.Figure(
            data=go.Heatmap(
                z=profile.correlation_matrix,
                x=profile.correlation_columns,
                y=profile.correlation_columns,
                colorscale="RdBu",
                zmid=0,
                hovertemplate="%{y} vs %{x}: %{z}<extra></extra>",
            )
        )
        corr_figure.update_layout(
            title="Correlation Heatmap",
            margin=dict(l=40, r=20, t=55, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
    else:
        corr_figure = go.Figure()
        corr_figure.update_layout(
            title="Correlation Heatmap",
            annotations=[dict(text="No numeric correlation matrix available", showarrow=False)],
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    missing_counts = frame.isna().sum()
    missing_figure = go.Figure(
        go.Bar(
            x=missing_counts.index.tolist(),
            y=missing_counts.values.tolist(),
            marker=dict(color="#f97316"),
        )
    )
    missing_figure.update_layout(
        title="Missing Values by Column",
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis_title="Column",
        yaxis_title="Missing count",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    selected_series = frame[active_column]
    if pd.api.types.is_numeric_dtype(selected_series):
        column_figure = go.Figure(go.Histogram(x=selected_series, marker=dict(color="#14b8a6")))
        column_figure.update_layout(
            title=f"Distribution of {active_column}",
            margin=dict(l=40, r=20, t=55, b=40),
            xaxis_title=active_column,
            yaxis_title="Count",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
    else:
        counts = selected_series.astype(str).value_counts().head(15)
        column_figure = go.Figure(
            go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker=dict(color="#14b8a6"))
        )
        column_figure.update_layout(
            title=f"Top values for {active_column}",
            margin=dict(l=40, r=20, t=55, b=40),
            xaxis_title=active_column,
            yaxis_title="Count",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    return {
        "summary": {
            "dataset": profile.dataset_label,
            "task_type": profile.task_type,
            "target_column": profile.target_column,
            "rows": profile.row_count,
            "columns": profile.column_count,
            "missing_values": profile.missing_values,
            "duplicate_rows": profile.duplicate_rows,
        },
        "available_columns": available_columns,
        "selected_column": active_column,
        "column_profile": _column_profile(frame, active_column),
        "warnings": profile.warnings,
        "dtype_summary": profile.dtype_summary,
        "class_balance": profile.class_balance,
        "charts": {
            "target_distribution": figure_to_dict(target_figure),
            "correlation_heatmap": figure_to_dict(corr_figure),
            "missing_values": figure_to_dict(missing_figure),
            "column_distribution": figure_to_dict(column_figure),
        },
    }