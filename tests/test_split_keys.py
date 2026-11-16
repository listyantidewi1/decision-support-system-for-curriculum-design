"""Tests for gold_standard.posting_keys — the employer-key layer.

Covers review finding F03 (a 1-2 character loose residual silently switched the
loose employer relation OFF while `employer_known` stayed True), the
"company recorded but the residual degenerated" vs "no company recorded"
distinction that F03 showed was collapsed into one empty string, and the
documented empty-employer-hash fork (distinct groups, never collapsed).

Run:
  .venv-ml\\Scripts\\python.exe tests\\test_split_keys.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_standard import LEGAL_FORM_TOKENS                              # noqa: E402
from gold_standard import posting_keys as keys_mod                       # noqa: E402
from gold_standard.groups import _equality_edges                         # noqa: E402
from gold_standard.posting_keys import (                                 # noqa: E402
    LOOSE_STATUSES, LOOSE_STATUSES_WITHOUT_HASH, LOOSE_STATUS_COMPANY_MISSING,
    LOOSE_STATUS_OK, LOOSE_STATUS_RESIDUAL_EMPTY, LOOSE_STATUS_SHORT,
    employer_loose_key, is_missing_hash, make_employer_hash_loose)

h = make_employer_hash_loose
k = employer_loose_key

# Real classes from the pool that the pre-fix rule failed on (review finding
# F03): both residuals are 2 characters, so both were returned as "".
IT_CLASS = ("IT Group Pte Ltd", "IT Group Indonesia", "IT Thailand")
BD_CLASS = ("BD", "BD Malaysia")


# ------------------------------------------------------- F03: short residuals
def test_short_residual_is_hashed_not_disabled():
    """The negative test F03 says was never written, in BOTH directions."""
    hashes = {h(c) for c in IT_CLASS}
    assert len(hashes) == 1, "the 'it' residual class must share ONE loose hash"
    assert "" not in hashes, "a 2-char residual must NOT disable the loose relation"
    assert {k(c)[1] for c in IT_CLASS} == {LOOSE_STATUS_SHORT}

    assert h(BD_CLASS[0]) == h(BD_CLASS[1]) != ""
    # and a DIFFERENT short class must not collide with it
    assert h("HP Group Pte Ltd") not in ("", h(BD_CLASS[0]))
    # one character is still a name, not a missing value
    assert h("K Group") != "" and k("K Group")[1] == LOOSE_STATUS_SHORT
    assert h("E Holdings") != "" and h("E Holdings") != h("K Group")


def test_short_residual_class_produces_real_group_edges():
    """The mechanism F03 broke: _equality_edges builds no edge from "".

    gold_standard.groups is owned elsewhere and is NOT modified; it is called
    here read-only, because the whole point of F03 is that an empty loose hash
    makes the loose relation invisible to exactly this function.
    """
    frame = pd.DataFrame({
        "posting_id": ["p1", "p2", "p3"],
        "employer_hash_loose": [h(c) for c in IT_CLASS],
    })
    assert _equality_edges(frame, "employer_hash_loose") == [("p1", "p2"), ("p1", "p3")]

    # control: the pre-fix return value ("" for every member) yields NO edges,
    # which is the silent switch-off the finding reproduced.
    disabled = frame.assign(employer_hash_loose="")
    assert _equality_edges(disabled, "employer_hash_loose") == []


def test_short_residual_does_not_borrow_a_longer_residuals_hash():
    assert h("BD") != h("BDO Ltd")
    assert h("IT Group Pte Ltd") != h("ITC Ltd")


# -------------------------------------- empty hash: the two reasons, distinct
def test_empty_loose_hash_has_an_explicit_reason_in_the_data():
    """"no company recorded" and "residual degenerated" must be TELLABLE APART.

    Both return an empty hash and downstream code reads empty as missing, so the
    reason cannot live in the hash — it lives in `employer_loose_status`.
    """
    assert k(None) == ("", LOOSE_STATUS_COMPANY_MISSING)
    assert k(float("nan")) == ("", LOOSE_STATUS_COMPANY_MISSING)
    assert k("") == ("", LOOSE_STATUS_COMPANY_MISSING)
    # a company IS recorded here; its whole name is legal-form tokens
    assert k("PT Tbk") == ("", LOOSE_STATUS_RESIDUAL_EMPTY)
    assert k("Ltd") == ("", LOOSE_STATUS_RESIDUAL_EMPTY)
    assert k("Global Holdings Indonesia") == ("", LOOSE_STATUS_RESIDUAL_EMPTY)
    # the distinction is real, not cosmetic
    assert k(None)[1] != k("PT Tbk")[1]


def test_hash_is_empty_exactly_when_the_status_says_so():
    probes = [None, float("nan"), "", "   ", "---", "nan", "None", "NA", "<NA>",
              "PT Tbk", "Ltd", "BD", "K Group", "PT Alpha Beta", "Alpha Beta Ltd",
              "AT&T Inc", "7-Eleven Pte Ltd", "Müller GmbH", "บริษัท เทสต์ จำกัด"]
    for company in probes:
        hash_, status = k(company)
        assert status in LOOSE_STATUSES, f"{company!r} -> unknown status {status!r}"
        assert (hash_ == "") == (status in LOOSE_STATUSES_WITHOUT_HASH), company


def test_all_legal_form_names_do_not_collapse_into_one_employer():
    """Why `residual_empty` stays empty: it must not become ONE employer."""
    names = ["PT Tbk", "Ltd", "Global Holdings Indonesia", "Pte Ltd Singapore"]
    assert all(h(n) == "" for n in names)
    frame = pd.DataFrame({"posting_id": [f"p{i}" for i in range(len(names))],
                          "employer_hash_loose": [h(n) for n in names]})
    assert _equality_edges(frame, "employer_hash_loose") == []


def test_missing_sentinel_company_strings_are_missing_not_a_value():
    """"nan" as a company string must not become the residual "nan".

    Otherwise every posting whose company round-tripped to the literal string
    "nan" would share one fabricated employer — the exact failure the module
    forbids for `employer_hash`.
    """
    for sentinel in ("nan", "NaN", "none", "None", "null", "NA", "<NA>", "  ", "-"):
        assert k(sentinel) == ("", LOOSE_STATUS_COMPANY_MISSING), sentinel
    assert all(is_missing_hash(v) for v in
               ("", "nan", "None", "NULL", "<NA>", "na", None, float("nan")))
    assert not is_missing_hash("9e36c5849d257dc0")


# ------------------------------------------- the documented empty-hash fork
def test_empty_employer_hash_policy_is_DISTINCT_not_collapsed():
    """The spec's fork, pinned: company-less postings stay in distinct groups.

    split_spec_v1 §3.3 ("treat '' / null / 'nan' as MISSING, never as a value")
    chooses DISTINCT over one giant unknown-employer group. The enforcement point
    is groups._equality_edges dropping "" before building edges; this test pins
    the choice so flipping it cannot happen silently. Measured on the shipped
    pool: 417 postings with no employer_hash sit in 386 distinct leakage groups.
    """
    frame = pd.DataFrame({
        "posting_id": ["p1", "p2", "p3", "p4"],
        "employer_hash": ["", "", "", "9e36c5849d257dc0"],
        "employer_hash_loose": ["", "", "", "c534b96e02df5c88"],
    })
    assert _equality_edges(frame, "employer_hash") == []
    assert _equality_edges(frame, "employer_hash_loose") == []
    # and a real shared hash DOES build an edge, so the test above is not vacuous
    shared = frame.assign(employer_hash=["x", "x", "", "y"])
    assert _equality_edges(shared, "employer_hash") == [("p1", "p2")]


# ----------------------------------------------- normalisation and stability
def test_legal_form_stripping_keeps_token_order_and_ascii_hashes_stable():
    assert h("PT Alpha Beta") == h("Alpha Beta Ltd") == h("Alpha Beta Indonesia")
    assert h("Alpha Beta") != h("Beta Alpha")   # sorting would merge these

    # The Unicode-aware normalisation must not RE-KEY ascii-only names: the
    # 18,020 pool postings with a 3+ char ascii residual keep their hash.
    ascii_re = re.compile(r"[^0-9a-z]+")

    def legacy(company: str) -> str:
        text = ascii_re.sub(" ", unicodedata.normalize("NFKC", company).casefold())
        residual = " ".join(t for t in text.split() if t not in LEGAL_FORM_TOKENS)
        return "" if len(residual) < 3 else keys_mod._sha1_16(
            f"{keys_mod.EMPLOYER_HASH_SALT}|{residual}")

    for company in ("PT Alpha Beta", "Alpha-Beta Ltd.", "ACME Corp", "AT&T Inc",
                    "7-Eleven Pte Ltd", "Foo_Bar Holdings", "Bank BRI (Persero)"):
        assert h(company) == legacy(company) != "", company


def test_non_latin_company_name_is_not_deleted_by_normalisation():
    """82 of the 84 empty residuals in the real pool were non-Latin names.

    The old ascii-only `[^0-9a-z]+` rule removed every character of a Thai, CJK,
    Katakana or Hangul company name, so the residual was empty and the loose
    relation was off — reported as `residual_empty`, which was not the truth.
    """
    for company in ("บริษัท เทสต์ จำกัด", "株式会社テスト", "테스트 주식회사"):
        hash_, status = k(company)
        assert hash_ != "", company
        assert status == LOOSE_STATUS_OK, (company, status)

    # Thai vowel signs are Unicode category Mn: a `\W`-based rule would split the
    # word at every mark. Same name with different punctuation/spacing must hash
    # the same.
    assert h("บริษัท เทสต์ จำกัด") == h("บริษัท  (เทสต์)  จำกัด")
    assert h("株式会社テスト") != h("株式会社テスト2")
    # a diacritic is part of the name, not noise to be dropped
    assert h("Müller GmbH") != "" and h("Müller GmbH") != h("M Ller GmbH")


# ------------------------------------------------- end-to-end over a tiny build
def _tiny_build(tmp: Path) -> tuple[pd.DataFrame, dict]:
    """Run build_posting_keys over a 7-posting synthetic raw batch."""
    rows = [
        # posting, company, employer_hash  -> expected loose status
        ("u-ok1", "PT Alpha Beta", "e-alpha"),            # ok
        ("u-ok2", "Alpha Beta Ltd", "e-alpha2"),          # ok, same loose class
        ("u-sh1", "IT Group Pte Ltd", "e-it1"),           # short_residual
        ("u-sh2", "IT Thailand", "e-it2"),                # short_residual, same
        ("u-re1", "PT Tbk", "e-re1"),                     # residual_empty
        ("u-cm1", None, None),                            # company_missing
        ("u-cm2", "", ""),                                # company_missing
    ]
    raw = pd.DataFrame({
        "source_record_id": [r[0] for r in rows],
        "acquisition_timestamp": ["2026-01-01T00:00:00Z"] * len(rows),
        "company": [r[1] for r in rows],
        "employer_hash": [r[2] for r in rows],
        "query_family": ["software_developer"] * len(rows),
        "date_posted": ["2026-01-02"] * len(rows),
    })
    batch = tmp / "corpus_raw" / "batch-000"
    batch.mkdir(parents=True)
    raw.to_parquet(batch / "records.parquet", index=False)

    flags = pd.DataFrame({
        "source_record_id": [r[0] for r in rows],
        "stratum": ["indonesia"] * len(rows),
        "country_iso": ["ID"] * len(rows),
        "site": ["jobstreet_id"] * len(rows),
        "eligible": ["True"] * len(rows),
    })
    flags_path = tmp / "record_flags_20260101T000000Z.parquet"
    flags.to_parquet(flags_path, index=False)

    pool_path = tmp / "sentence_pool.parquet"
    pd.DataFrame({"posting_id": [r[0] for r in rows[:6]]}).to_parquet(
        pool_path, index=False)      # u-cm2 deliberately out of pool

    out = tmp / "gs"
    frame = keys_mod.build_posting_keys(raw_root=tmp / "corpus_raw",
                                       flags_path=flags_path, pool_path=pool_path,
                                       out_dir=out, verbose=False)
    manifest = json.loads((out / "posting_keys_manifest.json").read_text(encoding="utf-8"))
    return frame, manifest


def test_built_frame_carries_the_status_column():
    with tempfile.TemporaryDirectory(prefix="pkeys_") as td:
        frame, _ = _tiny_build(Path(td))
    assert "employer_loose_status" in frame.columns
    status = dict(zip(frame["posting_id"], frame["employer_loose_status"]))
    assert status == {"u-ok1": LOOSE_STATUS_OK, "u-ok2": LOOSE_STATUS_OK,
                      "u-sh1": LOOSE_STATUS_SHORT, "u-sh2": LOOSE_STATUS_SHORT,
                      "u-re1": LOOSE_STATUS_RESIDUAL_EMPTY,
                      "u-cm1": LOOSE_STATUS_COMPANY_MISSING,
                      "u-cm2": LOOSE_STATUS_COMPANY_MISSING}
    loose = dict(zip(frame["posting_id"], frame["employer_hash_loose"]))
    # the short class is a REAL class: one shared, non-empty hash
    assert loose["u-sh1"] == loose["u-sh2"] != ""
    assert loose["u-ok1"] == loose["u-ok2"] != ""
    assert loose["u-re1"] == loose["u-cm1"] == loose["u-cm2"] == ""
    # ... and employer_known is unchanged by any of this
    known = dict(zip(frame["posting_id"], frame["employer_known"]))
    assert known["u-sh1"] and known["u-re1"] and not known["u-cm1"]
    # every value is a member of the closed set
    assert set(frame["employer_loose_status"]) <= set(LOOSE_STATUSES)


def test_manifest_reports_the_status_counts_and_the_empty_key_policy():
    with tempfile.TemporaryDirectory(prefix="pkeys_") as td:
        frame, manifest = _tiny_build(Path(td))
    block = manifest["employer_hash_loose"]
    assert block["status_counts_eligible"] == {
        LOOSE_STATUS_OK: 2, LOOSE_STATUS_SHORT: 2,
        LOOSE_STATUS_RESIDUAL_EMPTY: 1, LOOSE_STATUS_COMPANY_MISSING: 2}
    # in-pool counts are the ones the group layer sees (u-cm2 is out of pool)
    assert block["status_counts_in_pool"][LOOSE_STATUS_COMPANY_MISSING] == 1
    assert block["status_counts_in_pool"][LOOSE_STATUS_SHORT] == 2
    # a regression in the F03 rule shows up as a NON-ZERO number here
    assert block["n_loose_disabled_by_short_residual"] == 0
    assert block["n_short_residual_hashed"] == 2
    assert block["n_nonempty"] == 4 and block["n_nonempty_in_pool"] == 4
    assert "deviation_from_split_spec_v1" in block

    policy = manifest["empty_key_policy"]
    assert policy["choice"] == "distinct"
    assert policy["n_without_employer_hash"] == 2
    assert policy["n_without_employer_hash_in_pool"] == 1
    assert policy["n_without_loose_hash_in_pool"] == 2    # u-re1 + u-cm1
    assert "groups._equality_edges" in policy["enforced_by"]


def test_build_asserts_hash_and_status_agree():
    """The build-time guard fires rather than writing an inconsistent frame."""
    original = keys_mod.employer_loose_key
    keys_mod.employer_loose_key = lambda company: ("", LOOSE_STATUS_OK)
    try:
        with tempfile.TemporaryDirectory(prefix="pkeys_") as td:
            try:
                _tiny_build(Path(td))
            except ValueError as exc:
                assert "disagree" in str(exc)
            else:
                raise AssertionError("an empty hash with status 'ok' must raise")
    finally:
        keys_mod.employer_loose_key = original


def _run_standalone() -> int:
    """Plain-python runner (the repo has no pytest); pytest-compatible too."""
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
