"""Append-only raw batch store: DATA/corpus_raw/<batch_id>/records.parquet + manifest.json.

Immutability contract: a batch directory with records in it is never rewritten.
The registry (one row per record per batch) is derived data, rebuildable from the
batches, and is what first_seen/last_seen are computed from (proposal Tabel 3.10).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import HARNESS_VERSION
from .schema import SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "DATA" / "corpus_raw"

REGISTRY_COLUMNS = [
    "source_record_id", "batch_id", "acquisition_timestamp",
    "date_posted", "site", "country_iso", "stratum",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet-safe copy: mixed-type object columns become nullable strings."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out


def write_batch(
    df: pd.DataFrame,
    batch_id: str,
    manifest_extra: dict | None = None,
    out_root: Path | str | None = None,
) -> Path:
    """Write one immutable batch (records + manifest) and update the registry."""
    root = Path(out_root) if out_root else DEFAULT_ROOT
    batch_dir = root / batch_id
    records_path = batch_dir / "records.parquet"
    if records_path.exists():
        raise FileExistsError(
            f"{records_path} already exists — batches are immutable. "
            f"Use a new batch_id instead of rewriting."
        )
    batch_dir.mkdir(parents=True, exist_ok=True)

    safe = _coerce_object_columns(df)
    safe.to_parquet(records_path, index=False)

    dates = pd.to_datetime(df.get("date_posted"), errors="coerce")
    manifest = {
        "batch_id": batch_id,
        "created_utc": _utc_now_iso(),
        "harness_version": HARNESS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_rows": int(len(df)),
        "n_unique_source_record_id": int(df["source_record_id"].nunique()),
        "rows_by_site": {str(k): int(v) for k, v in df["site"].value_counts().items()} if "site" in df else {},
        "rows_by_country": {str(k): int(v) for k, v in df["country_iso"].value_counts().items()},
        "rows_by_stratum": {str(k): int(v) for k, v in df["stratum"].value_counts().items()},
        "date_posted_min": str(dates.min()) if dates.notna().any() else None,
        "date_posted_max": str(dates.max()) if dates.notna().any() else None,
        "date_posted_null_pct": round(float(dates.isna().mean()), 4) if len(df) else None,
    }
    manifest.update(manifest_extra or {})
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    _update_registry(df, root)
    return batch_dir


def _update_registry(df: pd.DataFrame, root: Path) -> None:
    registry_path = root / "registry.parquet"
    cols = [c for c in REGISTRY_COLUMNS if c in df.columns]
    entry = df[cols].copy()
    entry["date_posted"] = entry.get("date_posted", pd.Series(index=entry.index)).astype(str)
    entry = _coerce_object_columns(entry)
    if registry_path.exists():
        entry = pd.concat([pd.read_parquet(registry_path), entry], ignore_index=True)
    entry.to_parquet(registry_path, index=False)


def seen_spans(out_root: Path | str | None = None) -> pd.DataFrame:
    """first_seen/last_seen per source_record_id across all batches (Tabel 3.10)."""
    root = Path(out_root) if out_root else DEFAULT_ROOT
    registry = pd.read_parquet(root / "registry.parquet")
    grouped = registry.groupby("source_record_id")["acquisition_timestamp"]
    return pd.DataFrame({
        "first_seen": grouped.min(),
        "last_seen": grouped.max(),
        "n_batches": grouped.count(),
    }).reset_index()
