"""Serve the annotation sample to the annotator UI, scoped to their token.

Why this exists: `DATA/gold_standard/` is gitignored, and the Streamlit UI is
deployed from the GitHub repo. A hosted UI therefore has NO sentence data on
disk and `load_sample()` would raise FileNotFoundError on first page load. The
collector already has a token per annotator and a persistent disk, so it is the
natural place to hold the sentences too.

Two consequences worth stating, because both are properties we want:
  * the UI holds no research data at all — it renders whatever the collector
    hands it, so nothing is published to GitHub and nothing is left behind on an
    ephemeral host;
  * sentence delivery is gated by the SAME token as writing, so a practice token
    receives practice sentences and the pilot sentences are not world-readable
    from the public UI URL.

CSV, not parquet, deliberately: the collector's whole dependency set is
fastapi + uvicorn, and reading parquet would pull in pandas and pyarrow for a
49 KB file. `gold_standard.sample` already writes both formats.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# Where the {split}.csv files live. The container bakes them in at /app/samples;
# a local run falls back to the repo layout so `uvicorn ... --reload` works with
# no configuration.
DEFAULT_SAMPLES_DIR = "samples"
_REPO_FALLBACK = Path(__file__).resolve().parent.parent / "DATA" / "gold_standard"


class SampleUnavailable(RuntimeError):
    """The split's CSV is absent — a deployment error, never a client error."""


def samples_dir() -> Path:
    """Resolved at CALL time so tests and redeploys can repoint it (the same
    mistake was made once with GS_DB_PATH and had to be fixed)."""
    env = os.environ.get("GS_SAMPLES_DIR")
    if env:
        return Path(env)
    d = Path(DEFAULT_SAMPLES_DIR)
    return d if d.is_dir() else _REPO_FALLBACK


def _split_path(split: str) -> Path:
    # Defend the path join: `split` arrives from a token, but a token file is
    # hand-edited and a stray "../" must not escape the samples directory.
    if not split or "/" in split or "\\" in split or split != Path(split).name \
            or split.startswith("."):
        raise SampleUnavailable(f"invalid split name: {split!r}")
    return samples_dir() / f"{split}.csv"


def load_rows(split: str) -> list[dict]:
    """Return the split's sentences in sample_order, as plain JSON-able dicts."""
    p = _split_path(split)
    if not p.is_file():
        raise SampleUnavailable(
            f"no sample file for split {split!r} at {p}. Run "
            "`python -m gold_standard.sample` and redeploy the collector so the "
            "CSV is baked into the image.")
    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SampleUnavailable(f"sample file for {split!r} is empty: {p}")

    def _order(r: dict) -> tuple:
        try:
            return (0, int(r.get("sample_order") or 0))
        except (TypeError, ValueError):
            return (1, 0)

    rows.sort(key=_order)
    return rows


def available_splits() -> list[str]:
    d = samples_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.csv"))
