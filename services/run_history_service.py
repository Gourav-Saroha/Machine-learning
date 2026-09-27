from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_HISTORY_PATH = PROJECT_ROOT / "data" / "run_history.json"


def _load_history() -> list[dict[str, Any]]:
    if not RUN_HISTORY_PATH.exists():
        return []
    return json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))


def _write_history(rows: list[dict[str, Any]]) -> None:
    RUN_HISTORY_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def save_run_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    rows = _load_history()
    record = {
        "id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_key": entry.get("dataset_key"),
        "dataset_label": entry.get("dataset_label"),
        "model_type": entry.get("model_type"),
        "hyperparameters": entry.get("hyperparameters", {}),
        "metrics": entry.get("metrics", {}),
        "notes": entry.get("notes", ""),
    }
    rows.insert(0, record)
    _write_history(rows[:200])
    return record


def load_run_history(dataset_key: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
    rows = _load_history()
    for row in rows:
        row.setdefault("id", str(uuid4()))
    if dataset_key:
        rows = [row for row in rows if row.get("dataset_key") == dataset_key]
    return rows[:limit]


def update_run_history_notes(run_id: str, notes: str) -> dict[str, Any]:
    rows = _load_history()
    for row in rows:
        row.setdefault("id", str(uuid4()))
        if row["id"] == run_id:
            row["notes"] = notes
            _write_history(rows)
            return row
    raise ValueError(f"Run '{run_id}' not found.")