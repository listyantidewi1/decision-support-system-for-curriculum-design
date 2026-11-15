"""Tests for the gold-standard sample pipeline (build_pool / sample / export_xlsx).

These three modules had NO test coverage, which is how several text-fidelity and
quota defects survived: the text annotators read, and that span offsets index
into, is produced here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard.build_pool import (  # noqa: E402
    MAX_SENT_CHARS, _html_to_text, _normalize_sentence, _resplit_long,
)
from gold_standard.export_xlsx import SPAN_ROWS_PER_SENTENCE, _maxrow  # noqa: E402
from gold_standard.sample import _spread_sample  # noqa: E402


# ------------------------------------------------------------------ build_pool
def test_html_entities_resolved_before_tag_strip():
    # single- and double-escaped HTML previously survived into the sentence text
    assert "<p>" not in _html_to_text("&lt;p&gt;Menguasai SQL&lt;/p&gt;")
    assert "&lt;" not in _html_to_text("&amp;lt;p&amp;gt;Menguasai SQL&amp;lt;/p&amp;gt;")
    assert "Menguasai SQL" in _html_to_text("&lt;p&gt;Menguasai SQL&lt;/p&gt;")


def test_html_amp_entity_decoded():
    assert "R&D" in _html_to_text("Divisi R&amp;D")


def test_plus_and_numbered_bullets_become_boundaries():
    # '+' bullets and '1.'/'2)' markers were handled nowhere, fusing list items
    plus = _html_to_text("+ Menguasai SQL\n+ Menguasai Docker")
    assert plus.count("•") == 2, plus
    num = _html_to_text("1. Menguasai SQL\n2) Menguasai Docker")
    assert num.count("•") == 2, num


def test_normalize_sentence_collapses_whitespace_and_bullets():
    assert _normalize_sentence("  •  Menguasai   SQL \n") == "Menguasai SQL"


def test_resplit_long_actually_splits_a_runon():
    # a run-on with no terminal punctuation, joined by spaced dashes + bullets
    runon = _normalize_sentence(
        " - ".join(["Menguasai pemrograman berorientasi objek dan basis data"] * 8))
    assert len(runon) > MAX_SENT_CHARS
    pieces = _resplit_long(runon)
    assert len(pieces) > 1
    assert all(p.strip() for p in pieces)


def test_resplit_long_leaves_short_sentences_alone():
    s = "Menguasai SQL."
    assert _resplit_long(s) == [s]


# ---------------------------------------------------------------- export_xlsx
def test_validation_range_covers_the_whole_main_sample():
    """A fixed 5000-row validation range left most of the main-1,200 workbook
    (1 + 1200*15 = 18,001 rows) without locked dropdowns."""
    rows_needed = 1 + 1200 * (SPAN_ROWS_PER_SENTENCE + 1)
    assert _maxrow(1200) >= rows_needed
    assert _maxrow(100) >= 1 + 100 * (SPAN_ROWS_PER_SENTENCE + 1)


# --------------------------------------------------------------------- sample
def _pool(n, sites=("indeed", "glints"), bands=("short", "medium")):
    return pd.DataFrame({
        "sentence_id": [f"s{i}" for i in range(n)],
        "posting_id": [f"p{i // 2}" for i in range(n)],
        "site": [sites[i % len(sites)] for i in range(n)],
        "length_band": [bands[i % len(bands)] for i in range(n)],
    })


def test_spread_sample_never_exceeds_k():
    df = _pool(50)
    for k in (1, 7, 13, 49):
        assert len(_spread_sample(df, k, seed=1)) <= k


def test_spread_sample_returns_all_when_k_exceeds_pool():
    df = _pool(5)
    assert len(_spread_sample(df, 99, seed=1)) == 5


def test_spread_sample_is_deterministic_for_a_seed():
    df = _pool(40)
    a = _spread_sample(df, 11, seed=42)["sentence_id"].tolist()
    b = _spread_sample(df, 11, seed=42)["sentence_id"].tolist()
    assert a == b


def test_spread_sample_empty_and_zero():
    assert _spread_sample(_pool(10), 0, seed=1).empty
    assert _spread_sample(_pool(0), 5, seed=1).empty


def _run_standalone() -> int:
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
