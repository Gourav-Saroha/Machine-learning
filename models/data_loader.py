from __future__ import annotations

import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
HEART_DATASET_PATH = DATA_DIR / "heart_v2.csv"
UPLOADS_DIR = DATA_DIR / "uploads"
DATASET_REGISTRY_PATH = DATA_DIR / "dataset_registry.json"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetProfile:
    dataset_key: str
    dataset_label: str
    source_type: str
    row_count: int
    column_count: int
    target_column: str
    task_type: str
    missing_values: int
    duplicate_rows: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    class_balance: list[dict[str, Any]]
    dtype_summary: list[dict[str, Any]]
    target_distribution: list[dict[str, Any]]
    correlation_matrix: list[list[float]]
    correlation_columns: list[str]
    warnings: list[str]


@dataclass
class DatasetDefinition:
    key: str
    label: str
    source_type: str
    file_path: str | None = None
    target_column: str | None = None
    task_type: str = "classification"


@dataclass
class DatasetBundle:
    dataset_key: str
    dataset_label: str
    source_type: str
    target_column: str
    task_type: str
    original_frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    target_names: list[str]
    profile: DatasetProfile


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "dataset"


def _registry_entries() -> dict[str, dict[str, Any]]:
    if not DATASET_REGISTRY_PATH.exists():
        return {}
    return json.loads(DATASET_REGISTRY_PATH.read_text(encoding="utf-8"))


def _write_registry(entries: dict[str, dict[str, Any]]) -> None:
    DATASET_REGISTRY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def dataset_columns(dataset_key: str) -> list[str]:
    definition = _dataset_definition(dataset_key)
    if definition.key == "iris":
        dataset = load_iris(as_frame=True)
        return list(dataset.frame.columns) + ["target"]
    if not definition.file_path:
        raise FileNotFoundError(f"Dataset '{dataset_key}' does not have a file path.")
    return pd.read_csv(definition.file_path, nrows=1).columns.tolist()


def _base_definitions() -> dict[str, DatasetDefinition]:
    return {
        "iris": DatasetDefinition(key="iris", label="Iris", source_type="builtin", task_type="classification"),
        "heart": DatasetDefinition(
            key="heart",
            label="Heart Disease CSV",
            source_type="project",
            file_path=str(HEART_DATASET_PATH),
            target_column="heart disease",
            task_type="classification",
        ),
    }


def _dataset_definition(dataset_key: str) -> DatasetDefinition:
    normalized_key = dataset_key.strip().lower()
    base = _base_definitions()
    if normalized_key in base:
        return base[normalized_key]

    registry = _registry_entries()
    if normalized_key in registry:
        item = registry[normalized_key]
        return DatasetDefinition(**item)

    raise ValueError(f"Unsupported dataset '{dataset_key}'.")


def _detect_target_column(frame: pd.DataFrame) -> str:
    preferred_names = ["target", "label", "class", "y", "outcome", "heart disease", "diagnosis"]
    lower_map = {column.lower(): column for column in frame.columns}
    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]
    return frame.columns[-1]


def _infer_task_type(target: pd.Series) -> str:
    unique_count = target.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(target) and unique_count > 20:
        return "regression"
    return "classification"


def _prepare_features(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    y = frame[target_column].copy()
    X = frame.drop(columns=[target_column]).copy()

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].fillna(X[column].median())
        else:
            X[column] = X[column].fillna("Missing")

    X = pd.get_dummies(X, drop_first=False)

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.Categorical(y).codes, index=y.index, name=target_column)

    return X, y.astype(int) if _infer_task_type(y) == "classification" else y, list(X.columns)


def _target_names(raw_target: pd.Series, task_type: str) -> list[str]:
    if task_type == "classification":
        categories = pd.Series(raw_target).astype("category").cat.categories.tolist()
        return [str(category) for category in categories]
    return [raw_target.name or "target"]


def _class_balance(raw_target: pd.Series, task_type: str) -> list[dict[str, Any]]:
    if task_type != "classification":
        return []
    counts = raw_target.astype("category").value_counts(dropna=False)
    total = int(counts.sum())
    return [
        {
            "label": str(label),
            "count": int(count),
            "pct": round(float(count / total * 100), 2),
        }
        for label, count in counts.items()
    ]


def _dtype_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "dtype": str(frame[column].dtype),
            "missing": int(frame[column].isna().sum()),
        }
        for column in frame.columns
    ]


def _target_distribution(raw_target: pd.Series, task_type: str) -> list[dict[str, Any]]:
    if task_type == "classification":
        return _class_balance(raw_target, task_type)
    bins = pd.cut(raw_target, bins=min(10, raw_target.nunique()), duplicates="drop").value_counts().sort_index()
    return [{"label": str(label), "count": int(count)} for label, count in bins.items()]


def _correlation_payload(frame: pd.DataFrame, target_column: str) -> tuple[list[list[float]], list[str]]:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if target_column not in numeric.columns:
        return [], []
    limited = numeric.iloc[:, : min(8, len(numeric.columns))]
    return limited.corr().fillna(0.0).round(4).values.tolist(), list(limited.columns)


def dataset_profile(dataset_key: str) -> DatasetProfile:
    definition = _dataset_definition(dataset_key)
    if definition.key == "iris":
        dataset = load_iris(as_frame=True)
        raw_frame = dataset.frame.copy()
        raw_frame["target"] = dataset.target
        target_column = "target"
        task_type = "classification"
        label = definition.label
    else:
        if not definition.file_path:
            raise FileNotFoundError(f"Dataset '{dataset_key}' does not have a file path.")
        raw_frame = pd.read_csv(definition.file_path)
        target_column = definition.target_column or _detect_target_column(raw_frame)
        task_type = definition.task_type or _infer_task_type(raw_frame[target_column])
        label = definition.label

    numeric_columns = raw_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in raw_frame.columns if column not in numeric_columns]
    warnings: list[str] = []
    if raw_frame.isna().sum().sum() > 0:
        warnings.append("Missing values detected; numeric columns are median-filled and categoricals use 'Missing'.")
    if task_type != "classification":
        warnings.append("Current ensemble lab is optimized for classification datasets. Regression datasets may need adapted pages.")
    if len(categorical_columns) > 0:
        warnings.append("Categorical columns are one-hot encoded automatically.")

    correlation_matrix, correlation_columns = _correlation_payload(raw_frame, target_column)

    return DatasetProfile(
        dataset_key=definition.key,
        dataset_label=label,
        source_type=definition.source_type,
        row_count=int(raw_frame.shape[0]),
        column_count=int(raw_frame.shape[1]),
        target_column=target_column,
        task_type=task_type,
        missing_values=int(raw_frame.isna().sum().sum()),
        duplicate_rows=int(raw_frame.duplicated().sum()),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        class_balance=_class_balance(raw_frame[target_column], task_type),
        dtype_summary=_dtype_summary(raw_frame),
        target_distribution=_target_distribution(raw_frame[target_column], task_type),
        correlation_matrix=correlation_matrix,
        correlation_columns=correlation_columns,
        warnings=warnings,
    )


def available_datasets() -> list[dict[str, str]]:
    definitions = list(_base_definitions().values())
    definitions.extend(DatasetDefinition(**item) for item in _registry_entries().values())
    return [
        {
            "key": definition.key,
            "label": definition.label,
            "source_type": definition.source_type,
        }
        for definition in definitions
    ]


def save_uploaded_dataset(file_name: str, file_bytes: bytes) -> dict[str, str]:
    safe_name = Path(file_name).name
    stem = _slugify(Path(safe_name).stem)
    target_path = UPLOADS_DIR / f"{stem}.csv"
    index = 1
    while target_path.exists():
        target_path = UPLOADS_DIR / f"{stem}-{index}.csv"
        index += 1

    target_path.write_bytes(file_bytes)
    frame = pd.read_csv(target_path)
    target_column = _detect_target_column(frame)
    task_type = _infer_task_type(frame[target_column])
    dataset_key = target_path.stem.replace("_", "-")
    label = Path(file_name).stem

    registry = _registry_entries()
    registry[dataset_key] = asdict(
        DatasetDefinition(
            key=dataset_key,
            label=label,
            source_type="upload",
            file_path=str(target_path),
            target_column=target_column,
            task_type=task_type,
        )
    )
    _write_registry(registry)
    return {"dataset_key": dataset_key, "dataset_label": label}


def update_dataset_target_column(dataset_key: str, target_column: str) -> dict[str, str]:
    definition = _dataset_definition(dataset_key)
    if definition.key == "iris":
        raise ValueError("The built-in Iris dataset uses a fixed target column.")
    if not definition.file_path:
        raise FileNotFoundError(f"Dataset '{dataset_key}' does not have a file path.")

    frame = pd.read_csv(definition.file_path, nrows=5)
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in dataset '{dataset_key}'.")

    registry = _registry_entries()
    registry[definition.key] = asdict(
        DatasetDefinition(
            key=definition.key,
            label=definition.label,
            source_type=definition.source_type,
            file_path=definition.file_path,
            target_column=target_column,
            task_type=definition.task_type,
        )
    )
    _write_registry(registry)
    return {"dataset_key": definition.key, "target_column": target_column}


def load_dataset(dataset_key: str = "iris") -> tuple[str, str, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load one of the supported datasets as pandas objects."""
    definition = _dataset_definition(dataset_key)
    if definition.key == "iris":
        dataset = load_iris(as_frame=True)
        X = dataset.data.copy()
        y = dataset.target.copy()
        return "iris", "Iris", X, y, list(dataset.feature_names), list(dataset.target_names)

    if not definition.file_path:
        raise FileNotFoundError(f"Dataset '{dataset_key}' does not have a file path.")
    frame = pd.read_csv(definition.file_path)
    target_column = definition.target_column or _detect_target_column(frame)
    task_type = definition.task_type or _infer_task_type(frame[target_column])
    X, y, feature_columns = _prepare_features(frame, target_column)
    target_names = _target_names(frame[target_column], task_type)
    return definition.key, definition.label, X, y, feature_columns, target_names


def load_dataset_bundle(dataset_key: str = "iris", random_state: int = 42, test_size: float = 0.2) -> DatasetBundle:
    """Create the shared train/test split used across the application."""
    definition = _dataset_definition(dataset_key)
    profile = dataset_profile(dataset_key)
    raw_frame = pd.read_csv(definition.file_path) if definition.file_path else load_iris(as_frame=True).frame.copy()
    dataset_key, dataset_label, X, y, feature_names, target_names = load_dataset(dataset_key)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if profile.task_type == "classification" else None,
    )
    return DatasetBundle(
        dataset_key=dataset_key,
        dataset_label=dataset_label,
        source_type=definition.source_type,
        target_column=profile.target_column,
        task_type=profile.task_type,
        original_frame=raw_frame,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        profile=profile,
    )


def dataset_summary(dataset_key: str = "iris") -> dict[str, int | str]:
    """Return dataset-level summary values for the overview dashboard."""
    profile = dataset_profile(dataset_key)
    return {
        "dataset": profile.dataset_label,
        "dataset_key": dataset_key,
        "samples": profile.row_count,
        "features": profile.column_count - 1,
        "classes": len(profile.class_balance) if profile.task_type == "classification" else 1,
        "target_column": profile.target_column,
        "task_type": profile.task_type,
    }
