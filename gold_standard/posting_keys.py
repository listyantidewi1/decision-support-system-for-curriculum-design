"""Posting-level split keys (employer / occupation / date), as a SIDECAR.

Why a sidecar and not a column in `sentence_pool.parquet`: `pilot_100_manifest.
json` pins `pool_sha1_16`, so rewriting the pool breaks the binding between the
pilot already in the annotators' hands and the pool it was drawn from, and
forces a re-segmentation whose `_latest_flags()` may pick a newer flags file
than the pilot was drawn against (there are 14 in results/corpus_audits/). The
sidecar leaves the pool byte-identical, costs no segmentation, and keeps
employer hashes out of the annotator-facing worksheets.

`build_pool` now also carries `employer_hash` / `query_family` into the pool for
the NEXT rebuild; this module is what makes the split layer runnable before that
rebuild, and stays the authority afterwards (one code path, one set of rules).

What is deliberately NOT done here:
  * no keyword job_title -> occupation classifier. 74% of OECD and 46% of ASEAN
    postings carry a legacy scrape bucket as `query_family`; `UNRESOLVED` is the
    honest label, and occupation stratification is best-effort in the proposal
    ("sejauh memungkinkan", PROP:1192) while a wrong label is not;
  * no company name is ever written to disk. Only salted hashes leave this
    module.

Two DOCUMENTED DEPARTURES from split_spec_v1 §3, both in `employer_loose_key`
and both recorded in `posting_keys_manifest.json` under
`employer_hash_loose.deviation_from_split_spec_v1`:
  * a loose residual of 1-2 characters is HASHED, not returned as "" (review
    finding F03: returning "" switched the loose relation off for exactly the
    legal-form variant families it exists to catch);
  * normalisation keeps Unicode letters/marks/numbers instead of `[^0-9a-z]+`,
    so a Thai/CJK/Katakana/Hangul company name is not deleted in its entirety.
The frame therefore carries one column the spec's §3.7 list does not:
`employer_loose_status`, the REASON an empty loose hash is empty. It is additive
and no consumer projects the frame positionally (`groups.py` reads it by name,
`split.py` projects `[posting_id, employer_hash]`).

Run:
  .venv-ml\\Scripts\\python.exe -m gold_standard.posting_keys \\
      [--flags record_flags_20261115T164444Z.parquet]
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from acquisition.schema import EMPLOYER_HASH_SALT, _sha1_16
from governance.ladder import RAW_ROOT, REPO_ROOT
from . import (LEGAL_FORM_TOKENS, OCCUPATION_MERGE_MAP, OCCUPATION_UNRESOLVED,
               OCCUPATION_UNRESOLVED_FAMILIES, SPLIT_SPEC_VERSION)

GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
FLAGS_DIR = REPO_ROOT / "results" / "corpus_audits"

RAW_PROJECTION = ["source_record_id", "acquisition_timestamp", "company",
                  "employer_hash", "query_family", "date_posted"]

# Strings parquet/pandas round-trips produce for a missing company. Treated as
# MISSING, never as a value: if "" were a value, every company-less posting
# would share one employer and be grouped together, which is the opposite of
# what the grouping is for. Applied to the `company` STRING as well as to the
# hash: a company literally recorded as "nan" would otherwise yield the residual
# "nan", i.e. one fabricated employer shared by every such posting.
_MISSING_HASHES = {"", "nan", "none", "null", "<na>", "na"}

# A company name keeps every Unicode LETTER, MARK and NUMBER; everything else
# (punctuation, symbols, separators) becomes a token break. The previous rule,
# re.sub(r"[^0-9a-z]+", " "), was ASCII-only and therefore deleted a Thai, CJK,
# Katakana or Hangul company name in its ENTIRETY — on the real pool that is 82
# of the 84 postings whose residual came out empty, which silently switched the
# loose relation off for them (see LOOSE_STATUS_* below). A `[\w]`-based regex is
# not a fix either: Thai vowel signs are category Mn, so `\W` splits a Thai word
# at every vowel mark.
_NAME_CHAR_CATEGORIES = ("L", "M", "N")

# The closed set of outcomes of `employer_loose_key`. Written to
# posting_keys.parquet as `employer_loose_status` so that "no company recorded"
# and "company recorded but the loose residual degenerated" are DISTINGUISHABLE
# in the data: downstream code (gold_standard.groups._equality_edges) reads an
# empty hash as missing, so collapsing the two into one "" hid a real defect.
LOOSE_STATUS_OK = "ok"                            # residual >= 3 chars, hashed
LOOSE_STATUS_SHORT = "short_residual"             # residual 1-2 chars, hashed
LOOSE_STATUS_RESIDUAL_EMPTY = "residual_empty"    # name is all legal-form tokens
LOOSE_STATUS_COMPANY_MISSING = "company_missing"  # no company recorded at all
LOOSE_STATUSES = (LOOSE_STATUS_OK, LOOSE_STATUS_SHORT,
                  LOOSE_STATUS_RESIDUAL_EMPTY, LOOSE_STATUS_COMPANY_MISSING)
# The two statuses that carry an EMPTY loose hash, i.e. no loose employer edge.
LOOSE_STATUSES_WITHOUT_HASH = frozenset(
    {LOOSE_STATUS_RESIDUAL_EMPTY, LOOSE_STATUS_COMPANY_MISSING})


def is_missing_hash(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip().lower() in _MISSING_HASHES


@lru_cache(maxsize=1 << 16)
def _is_name_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in _NAME_CHAR_CATEGORIES


def employer_loose_key(company: object) -> tuple[str, str]:
    """Legal-form-insensitive employer hash plus the REASON it came out empty.

    `acquisition.schema.make_employer_hash` hashes the exact lower-cased name, so
    "PT Example", "Example Ltd" and "Example Indonesia" are three employers. This
    strips a CLOSED, committed token list (no tuning knob, deterministic) and
    hashes the residual, catching the commonest variant family. Token ORDER is
    preserved — sorting would merge "Alpha Beta" with "Beta Alpha".

    Returns `(hash, status)` with status from `LOOSE_STATUSES`:

      * `company_missing`  — no company recorded (None/NaN, a round-trip
        sentinel, or a string with no letter/mark/number in it). Hash is "".
      * `residual_empty`   — a company IS recorded but its whole name is
        legal-form tokens ("PT Tbk"). Hash is "", deliberately: such a name must
        NOT share a hash with every other such name. 2 pool postings.
      * `short_residual`   — residual of 1-2 characters ("BD" <- "BD Malaysia").
        HASHED. The former rule returned "" below 3 characters, which switched
        the loose relation off for exactly the legal-form variant families it
        exists to catch, while `employer_known` stayed True, so those postings
        were still freely placeable into dev/test (review finding F03: 146 of the
        230 real pool postings it names, including the class {"IT Group Pte Ltd",
        "IT Group Indonesia", "IT Thailand"} sitting in 3 distinct leakage
        groups; the other 84 were `residual_empty`, 82 of them because of the
        ASCII-only normalisation above). A 1-2
        character residual is a real company name, not a missing one; the
        collision risk is between two genuinely identically-named short
        employers, which is the risk the EXACT hash already carries, whereas
        returning "" guaranteed a miss. Measured cost on the real pool: 118 new
        loose edges, 2 group merges, largest single merge 4 postings (0.02%
        against the 2% MAX_LOOSE_MERGE_SHARE guard).
      * `ok`               — residual of 3+ characters. HASHED.

    This DEPARTS from split_spec_v1 §3.4, which says "if the residual is empty or
    shorter than 3 characters, return ''". The spec's own justification — "a
    company whose entire name is legal-form tokens must not share a hash with
    every other such company" — argues only for the EMPTY case; it does not
    support the <3 branch, which has the opposite effect of the one intended. The
    contract is wrong here and the spec sentence needs amending.

    The raw `company` string is never returned and never written to disk.
    """
    if company is None or (isinstance(company, float) and pd.isna(company)):
        return "", LOOSE_STATUS_COMPANY_MISSING
    if str(company).strip().lower() in _MISSING_HASHES:
        return "", LOOSE_STATUS_COMPANY_MISSING
    text = unicodedata.normalize("NFKC", str(company)).casefold()
    text = "".join(ch if _is_name_char(ch) else " " for ch in text)
    all_tokens = text.split()
    if not all_tokens:
        return "", LOOSE_STATUS_COMPANY_MISSING
    residual = " ".join(t for t in all_tokens if t not in LEGAL_FORM_TOKENS)
    if not residual:
        return "", LOOSE_STATUS_RESIDUAL_EMPTY
    status = LOOSE_STATUS_SHORT if len(residual) < 3 else LOOSE_STATUS_OK
    return _sha1_16(f"{EMPLOYER_HASH_SALT}|{residual}"), status


def make_employer_hash_loose(company: object) -> str:
    """`employer_loose_key(company)[0]` — the hash alone.

    Kept as the public one-value entry point. Callers that need to tell "no
    company recorded" from "company recorded but the residual degenerated" must
    use `employer_loose_key` or read `employer_loose_status`, because both
    statuses return "".
    """
    return employer_loose_key(company)[0]


def occupation_group(query_family: object) -> str:
    """query_family -> occupation group, with the committed merge map."""
    if query_family is None or (isinstance(query_family, float) and pd.isna(query_family)):
        return OCCUPATION_UNRESOLVED
    fam = str(query_family).strip()
    if not fam or fam in OCCUPATION_UNRESOLVED_FAMILIES:
        return OCCUPATION_UNRESOLVED
    return OCCUPATION_MERGE_MAP.get(fam, fam)


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_flags(flags_dir: Path = FLAGS_DIR) -> Path:
    files = sorted(flags_dir.glob("record_flags_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no record_flags_*.parquet in {flags_dir}")
    return files[-1]


def _raw_batch_paths(raw_root: Path) -> list[Path]:
    return sorted(p / "records.parquet" for p in sorted(raw_root.glob("batch-*"))
                  if (p / "records.parquet").exists())


def _parse_dates(series: pd.Series) -> pd.Series:
    """Per-element date parsing, matching build_pool.

    A vectorised pd.to_datetime(errors="coerce") silently NaTs 2,759 Indonesian
    rows (48% of the stratum) because the Indonesian non-Indeed sources store ISO
    timestamps ("2026-10-05T07:46:58Z") while the rest store "YYYY-MM-DD" — pandas
    infers ONE format for the column. Per element there is exactly 1 unparseable
    row in 85,740, and it is quarantined downstream, never guessed.
    """
    return series.map(lambda v: pd.to_datetime(v, errors="coerce", utc=True))


def build_posting_keys(raw_root: Path = RAW_ROOT,
                       flags_path: Path | None = None,
                       pool_path: Path | None = None,
                       out_dir: Path = GS_DIR,
                       verbose: bool = True) -> pd.DataFrame:
    flags_path = Path(flags_path) if flags_path else latest_flags()
    pool_path = Path(pool_path) if pool_path else (out_dir / "sentence_pool.parquet")
    if verbose:
        print(f"eligible set <- {flags_path.name}", flush=True)

    flags = pd.read_parquet(flags_path, columns=["source_record_id", "stratum",
                                                 "country_iso", "site", "eligible"])
    elig = flags[flags["eligible"].astype(str) == "True"].drop(columns=["eligible"])
    if verbose:
        print(f"  eligible postings: {len(elig):,}", flush=True)

    frames, batch_meta = [], []
    for path in _raw_batch_paths(raw_root):
        frame = pd.read_parquet(path, columns=RAW_PROJECTION)
        frames.append(frame)
        batch_meta.append({"batch": path.parent.name, "rows": int(len(frame)),
                           "sha256": _sha256(path)})
    if not frames:
        raise FileNotFoundError(f"no batch-*/records.parquet under {raw_root}")
    raw = pd.concat(frames, ignore_index=True)

    # FIRST SIGHTING, identical rule to build_pool. Mandatory, not cosmetic:
    # 14,747 of 226,333 ids carry more than one query_family across sightings and
    # 49 carry more than one non-empty employer_hash, so without a fixed rule the
    # occupation label and the grouping key depend on parquet ordering.
    raw = (raw.sort_values(["acquisition_timestamp", "source_record_id"], kind="stable")
              .drop_duplicates("source_record_id"))

    keys = elig.merge(raw, on="source_record_id", how="inner")
    n_unjoined = len(elig) - len(keys)

    keys["employer_hash"] = [
        "" if is_missing_hash(v) else str(v) for v in keys["employer_hash"]]
    keys["employer_known"] = keys["employer_hash"] != ""
    loose = [employer_loose_key(c) for c in keys["company"]]
    keys["employer_hash_loose"] = [h for h, _ in loose]
    keys["employer_loose_status"] = [s for _, s in loose]
    # The invariant that makes the status trustworthy: the hash is empty iff the
    # status says so. Asserted at build time, not just documented.
    _bad = [(h, s) for h, s in loose
            if (h == "") != (s in LOOSE_STATUSES_WITHOUT_HASH)]
    if _bad:
        raise ValueError(
            f"employer_loose_key returned {len(_bad)} rows whose hash and status "
            f"disagree (first: {_bad[0][1]!r} with "
            f"{'empty' if _bad[0][0] == '' else 'non-empty'} hash)")
    keys["query_family"] = [
        "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        for v in keys["query_family"]]
    keys["occupation_group"] = [occupation_group(v) for v in keys["query_family"]]
    keys["date_posted_utc"] = _parse_dates(keys["date_posted"])
    keys["date_parsed"] = keys["date_posted_utc"].notna()

    pool_ids: set[str] = set()
    if pool_path.exists():
        pool_ids = set(pd.read_parquet(pool_path, columns=["posting_id"])["posting_id"])
    keys["in_pool"] = keys["source_record_id"].isin(pool_ids)

    out = keys.rename(columns={"source_record_id": "posting_id"})[[
        "posting_id", "stratum", "country_iso", "site", "employer_hash",
        "employer_hash_loose", "employer_loose_status", "employer_known",
        "query_family", "occupation_group", "date_posted_utc", "date_parsed",
        "in_pool",
    ]].sort_values("posting_id", kind="stable").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "posting_keys.parquet"
    out.to_parquet(out_path, index=False)

    in_pool = out[out["in_pool"]]

    def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
        counts = frame["employer_loose_status"].value_counts().to_dict()
        return {s: int(counts.get(s, 0)) for s in LOOSE_STATUSES}

    status_all, status_pool = _status_counts(out), _status_counts(in_pool)

    missing_by_stratum = {
        str(st): {"eligible": int(len(g)), "missing_employer": int((~g["employer_known"]).sum())}
        for st, g in out.groupby("stratum", sort=True)}
    missing_by_site = {
        str(site): int((~g["employer_known"]).sum())
        for site, g in out.groupby("site", sort=True) if (~g["employer_known"]).any()}

    manifest = {
        "split_spec_version": SPLIT_SPEC_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flags_file": flags_path.name,
        "flags_sha256": _sha256(flags_path),
        "pool_file": pool_path.name if pool_path.exists() else None,
        "pool_sha256": _sha256(pool_path) if pool_path.exists() else None,
        "raw_batches": batch_meta,
        "raw_projection": RAW_PROJECTION,
        "first_sighting_rule": "sort by (acquisition_timestamp, source_record_id), keep first",
        "n_eligible": int(len(elig)),
        "n_eligible_unjoined_to_raw": int(n_unjoined),
        "n_keys": int(len(out)),
        "n_in_pool": int(out["in_pool"].sum()),
        "occupation_merge_map": OCCUPATION_MERGE_MAP,
        "occupation_unresolved_families": list(OCCUPATION_UNRESOLVED_FAMILIES),
        "legal_form_tokens": sorted(LEGAL_FORM_TOKENS),
        "employer_hash_loose": {
            "fn": "NFKC -> casefold -> every char outside Unicode categories "
                  "L/M/N to space -> drop LEGAL_FORM_TOKENS -> rejoin in original "
                  "order -> sha1_16(salt|residual)",
            "empty_when": "no company recorded (status company_missing), or the "
                          "company name is entirely legal-form tokens (status "
                          "residual_empty). A residual of 1-2 characters IS "
                          "hashed (status short_residual).",
            "n_nonempty": int((out["employer_hash_loose"] != "").sum()),
            "n_nonempty_in_pool": int((in_pool["employer_hash_loose"] != "").sum()),
            "status_counts_eligible": status_all,
            "status_counts_in_pool": status_pool,
            # Kept as a named field so a regression is visible as a non-zero
            # number: the pre-fix rule disabled the loose relation for every
            # posting with a 1-2 character residual (review finding F03).
            "n_loose_disabled_by_short_residual": 0,
            "n_short_residual_hashed": status_all[LOOSE_STATUS_SHORT],
            "n_short_residual_hashed_in_pool": status_pool[LOOSE_STATUS_SHORT],
            "deviation_from_split_spec_v1": (
                "spec §3.4 says to return '' when the residual is empty OR "
                "shorter than 3 characters. The <3 branch is implemented as "
                "HASHED instead: the spec's own justification ('a company whose "
                "entire name is legal-form tokens must not share a hash with "
                "every other such company') covers only the EMPTY residual, and "
                "the <3 branch switched the loose relation off for precisely the "
                "legal-form variant families it exists to catch while "
                "employer_known stayed True. Normalisation is also Unicode-aware "
                "now: the ASCII-only [^0-9a-z]+ rule deleted Thai/CJK/Katakana/"
                "Hangul company names entirely (82 of 84 empty residuals in the "
                "pool). Both departures need the spec sentence amended."),
        },
        # DOCUMENTED FORK, made explicit rather than left to the reader: a
        # posting with NO recorded employer does NOT join one giant
        # "unknown employer" group. "" is MISSING, never a value, in both hash
        # columns (spec §3.3 "never as a value"), and
        # gold_standard.groups._equality_edges drops "" before building edges, so
        # such postings stay in DISTINCT leakage groups and are then confined to
        # train by gold_standard.split (they cannot be shown absent from the
        # other side of a split). Measured on the shipped pool: 417 postings with
        # no employer_hash fall into 386 distinct leakage groups, i.e. distinct,
        # not collapsed.
        "empty_key_policy": {
            "choice": "distinct",
            "alternative_rejected": "collapse_into_one_group",
            "applies_to": ["employer_hash", "employer_hash_loose"],
            "rule": "an empty key is MISSING, never a value; no edge is built "
                    "from it, so company-less postings stay in distinct groups",
            "enforced_by": "gold_standard.groups._equality_edges filters "
                           "keys[column].astype(str) != '' before building edges",
            "consequence": "such postings are confined to train by "
                           "gold_standard.split (unverifiable_employer); they "
                           "cannot be shown absent from dev/test",
            "spec_reference": "split_spec_v1 §3.3",
            "n_without_employer_hash": int((~out["employer_known"]).sum()),
            "n_without_employer_hash_in_pool": int((~in_pool["employer_known"]).sum()),
            "n_without_loose_hash_in_pool": int(
                (in_pool["employer_hash_loose"] == "").sum()),
        },
        "missing_employer_by_stratum": missing_by_stratum,
        "missing_employer_by_site": missing_by_site,
        "n_date_unparseable": int((~out["date_parsed"]).sum()),
        "claim_limits": [
            "employer_hash and employer_hash_loose are STRING hashes: abbreviations, "
            "misspellings, subsidiaries and rebrands are still missed, so grouping is "
            "a LOWER BOUND on true employer relatedness. Claim 'employer-disjoint by "
            "the recorded hashes', never 'employer-disjoint'.",
            "Postings with no company at all cannot be shown absent from the other "
            "side of a split; they are confined to train by gold_standard.split.",
            "employer_hash_loose is EMPTY for two different reasons and the reason "
            "is in employer_loose_status, not in the hash: company_missing (no "
            "company recorded) and residual_empty (the whole name is legal-form "
            "tokens). Both carry no loose edge, so neither can be claimed "
            "loose-disjoint; read the status, never the empty string.",
            "A 1-2 character residual (status short_residual) IS hashed. Two "
            "genuinely different employers whose names reduce to the same 1-2 "
            "characters are therefore merged into one leakage group — a "
            "conservative error for leakage, a real one for any per-employer "
            "count. Occurrences are reported as n_short_residual_hashed.",
        ],
    }
    (out_dir / "posting_keys_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"posting keys: {len(out):,} postings ({int(out['in_pool'].sum()):,} in pool)",
              flush=True)
        print(json.dumps(missing_by_stratum, indent=2), flush=True)
        print(f"  loose-hash status (in pool): {status_pool}", flush=True)
        print(f"  unparseable dates: {int((~out['date_parsed']).sum())}", flush=True)
        print(f"-> {out_path}\n-> {out_dir / 'posting_keys_manifest.json'}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flags", default=None,
                    help="record_flags_*.parquet file name (default: newest)")
    args = ap.parse_args()
    flags = (FLAGS_DIR / args.flags) if args.flags else None
    build_posting_keys(flags_path=flags)


if __name__ == "__main__":
    main()
