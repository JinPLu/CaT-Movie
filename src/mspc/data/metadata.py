"""Shared metadata validation for dataset manifests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


def require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_required_file_columns(value: Any, label: str = "metadata_filter.required_file_columns") -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    for column in value:
        require_non_empty_string(column, label)
    return value


def row_has_required_files(row: pd.Series, required_columns: list[str]) -> bool:
    for column in required_columns:
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            return False
        path = Path(str(value))
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def required_file_failure(row: dict[str, Any], required_columns: list[str]) -> dict[str, str]:
    failures: dict[str, str] = {}
    for column in required_columns:
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            failures[column] = "missing_value"
            continue
        path = Path(str(value))
        if not path.is_file():
            failures[column] = "missing_file"
        elif path.stat().st_size <= 0:
            failures[column] = "empty_file"
    return failures


def required_file_failures(
    metadata: pd.DataFrame,
    required_columns: list[str],
    *,
    num_workers: int,
) -> list[dict[str, str]]:
    rows = [metadata.iloc[index].to_dict() for index in range(len(metadata))]
    if num_workers <= 1 or len(rows) <= 1:
        return [required_file_failure(row, required_columns) for row in rows]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(lambda row: required_file_failure(row, required_columns), rows))


def filter_metadata_files(
    metadata: pd.DataFrame,
    *,
    metadata_path: str,
    metadata_filter: dict[str, Any],
    extra_required_columns: list[str] | tuple[str, ...] = (),
    summary_extra: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(metadata_filter, dict):
        raise ValueError("metadata_filter must be a dict")
    required_columns = validate_required_file_columns(metadata_filter["required_file_columns"])
    expected_source_rows = metadata_filter["expected_source_rows"]
    expected_rows = metadata_filter["expected_rows"]
    if "num_file_check_workers" not in metadata_filter:
        raise ValueError("metadata_filter.num_file_check_workers is required")
    num_file_check_workers = metadata_filter["num_file_check_workers"]
    if type(expected_source_rows) is not int or expected_source_rows < 0:
        raise ValueError("metadata_filter.expected_source_rows must be a non-negative int")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("metadata_filter.expected_rows must be a positive int")
    if type(num_file_check_workers) is not int or num_file_check_workers <= 0:
        raise ValueError("metadata_filter.num_file_check_workers must be a positive int")
    if len(metadata) != expected_source_rows:
        raise ValueError(f"{metadata_path}: expected {expected_source_rows} source rows, got {len(metadata)}")

    for column in [*required_columns, *extra_required_columns]:
        if column not in metadata.columns:
            raise ValueError(f"metadata is missing required column: {column}")

    dropped_by_column = {
        column: {"missing_value": 0, "missing_file": 0, "empty_file": 0}
        for column in required_columns
    }
    failures_by_row = required_file_failures(
        metadata,
        required_columns,
        num_workers=num_file_check_workers,
    )
    keep_mask = [not failures for failures in failures_by_row]
    for failures in failures_by_row:
        for column, reason in failures.items():
            dropped_by_column[column][reason] += 1

    filtered = metadata.loc[keep_mask].reset_index(drop=True)
    dropped_rows = int(len(metadata) - len(filtered))
    if len(filtered) != expected_rows:
        raise ValueError(f"{metadata_path}: expected {expected_rows} rows after filtering, got {len(filtered)}")
    if len(filtered) == 0:
        raise ValueError(f"{metadata_path}: all rows were filtered by metadata file checks")

    summary = {
        "metadata_path": str(metadata_path),
        "source_rows": int(len(metadata)),
        "kept_rows": int(len(filtered)),
        "dropped_rows": dropped_rows,
        "required_file_columns": required_columns,
        "num_file_check_workers": num_file_check_workers,
        "dropped_by_column": dropped_by_column,
    }
    if summary_extra:
        summary.update(summary_extra)
    return filtered, summary
