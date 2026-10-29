"""Denominator ladder (Gambar 3.7): N_raw -> N_valid_metadata -> N_unique -> N_domain -> N_eligible.

Reads every batch in DATA/corpus_raw/, applies the versioned Tabel 3.9 rules,
and writes an audit report (JSON + Markdown) plus per-record flags parquet to
results/corpus_audits/. Nothing in the raw layer is modified or deleted.

Run:  .venv-ml\\Scripts\\python.exe -m governance.ladder [--no-minhash] [--no-langdetect]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import GOVERNANCE_VERSION
from .rules import apply_rules, chunked_map, detect_languages, load_rules, normalized_text_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "DATA" / "corpus_raw"
OUT_ROOT = REPO_ROOT / "results" / "corpus_audits"

LOAD_COLUMNS = [
    "source_record_id", "batch_id", "stratum", "country_iso", "site",
    "date_posted", "acquisition_timestamp", "title", "description",
    "query_term", "query_family", "employer_hash",
]


def load_all_batches() -> pd.DataFrame:
    frames = []
    for batch_dir in sorted(RAW_ROOT.glob("batch-*")):
        path = batch_dir / "records.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=LOAD_COLUMNS))
            print(f"  loaded {batch_dir.name}: {len(frames[-1]):,} rows", flush=True)
    return pd.concat(frames, ignore_index=True)


def minhash_near_dups(texts: pd.Series, cfg: dict) -> pd.Series:
    """True for every record that is a near-duplicate of an earlier record."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=cfg["threshold"], num_perm=cfg["num_perm"])
    is_dup = pd.Series(False, index=texts.index)
    for n, (idx, text) in enumerate(texts.items(), 1):
        words = str(text)[: cfg["max_chars"]].lower().split()
        k = cfg["shingle_words"]
        shingles = {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 1))}
        mh = MinHash(num_perm=cfg["num_perm"])
        for s in shingles:
            mh.update(s.encode("utf-8"))
        if lsh.query(mh):
            is_dup[idx] = True
        else:
            lsh.insert(str(idx), mh)
        if n % 20000 == 0:
            print(f"  minhash {n:,}/{len(texts):,} ({int(is_dup.sum()):,} near-dups)", flush=True)
    return is_dup


def crosstab(df: pd.DataFrame, rows: str, cols: str) -> dict:
    t = df.groupby([rows, cols])["source_record_id"].nunique().unstack(fill_value=0)
    return {str(r): {str(c): int(v) for c, v in row.items()} for r, row in t.iterrows()}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="tabel_3_9_v1.json")
    ap.add_argument("--no-minhash", action="store_true")
    ap.add_argument("--no-langdetect", action="store_true")
    args = ap.parse_args()

    rules = load_rules(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading batches ...", flush=True)
    df = load_all_batches()
    n_raw = len(df)
    print(f"N_raw (all sightings): {n_raw:,}", flush=True)

    print("Applying Tabel 3.9 rules ...", flush=True)
    flags = apply_rules(df, rules)
    flags["meta_and_window"] = flags["meta_valid"] & flags["date_in_window"]
    n_valid = int(flags["meta_and_window"].sum())
    print(f"N_valid_metadata (incl. period rule): {n_valid:,}", flush=True)

    # N_unique: earliest sighting of each source_record_id among valid rows,
    # then near-exact (normalized-description hash) within that set.
    valid = df[flags["meta_and_window"]].copy()
    vflags = flags[flags["meta_and_window"]].copy()
    order = valid["acquisition_timestamp"].astype(str)
    first = ~valid.assign(_o=order).sort_values("_o")["source_record_id"].duplicated()
    first = first.reindex(valid.index, fill_value=False)
    reps = valid[first].copy()
    rflags = vflags[first].copy()
    n_unique_exact = len(reps)

    print("Near-exact dedup (normalized description hash) ...", flush=True)
    norm_hash = chunked_map(reps["description"], normalized_text_hash)
    norm_dup = norm_hash.duplicated() & (norm_hash != "")
    rflags["near_exact_dup"] = norm_dup
    n_unique_norm = int((~norm_dup).sum())
    print(f"N_unique: exact {n_unique_exact:,} -> near-exact {n_unique_norm:,}", flush=True)

    rflags["near_dup_minhash"] = False
    if not args.no_minhash:
        print("MinHash LSH near-duplicate pass ...", flush=True)
        pool = reps[~norm_dup & rflags["domain_ok"]]
        rflags.loc[pool.index, "near_dup_minhash"] = minhash_near_dups(
            pool["description"], rules["duplication"]["near_dup"]
        )
    unique_mask = ~norm_dup & ~rflags["near_dup_minhash"]
    n_unique = int(unique_mask.sum())
    print(f"N_unique (after MinHash): {n_unique:,}", flush=True)

    dom_mask = unique_mask & rflags["domain_ok"] & rflags["content_ok"]
    n_domain = int(dom_mask.sum())
    print(f"N_domain (Tabel 3.9 function rules): {n_domain:,}", flush=True)

    rflags["language"] = "skipped"
    rflags["language_allowed"] = True
    if not args.no_langdetect:
        print("Language detection on domain-passing records ...", flush=True)
        langs = detect_languages(reps.loc[dom_mask, "description"])
        rflags.loc[dom_mask, "language"] = langs
        rflags.loc[dom_mask, "language_allowed"] = langs.isin(rules["language"]["allowed"])

    elig_mask = dom_mask & rflags["language_allowed"]
    n_eligible = int(elig_mask.sum())
    elig = reps[elig_mask]
    n_elig_strict = int((~rflags.loc[elig_mask, "conditional_platform"]).sum())
    print(f"N_eligible: {n_eligible:,} (excl. conditional platforms per K5: {n_elig_strict:,})", flush=True)

    # Coverage gate + representativeness
    gate_min = rules["coverage_gate"]["min_unique_eligible_per_country"]
    per_country = elig.groupby("country_iso")["source_record_id"].nunique().sort_values(ascending=False)
    gates = {c: {"n_eligible": int(n), "passes_gate": bool(n >= gate_min)} for c, n in per_country.items()}

    elig_q = elig.assign(quarter=pd.to_datetime(elig["date_posted"], errors="coerce",
                         utc=True, format="mixed").dt.to_period("Q").astype(str))
    report = {
        "created_utc": stamp,
        "governance_version": GOVERNANCE_VERSION,
        "config_version": rules["config_version"],
        "config_file": args.config,
        "minhash": not args.no_minhash,
        "langdetect": not args.no_langdetect,
        "ladder": {
            "N_raw": n_raw,
            "N_valid_metadata": n_valid,
            "N_unique_exact": n_unique_exact,
            "N_unique_near_exact": n_unique_norm,
            "N_unique": n_unique,
            "N_domain": n_domain,
            "N_eligible": n_eligible,
            "N_eligible_excl_conditional_platforms": n_elig_strict,
        },
        "eligible_by_stratum": {k: int(v) for k, v in
                                elig.groupby("stratum")["source_record_id"].nunique().items()},
        "coverage_gate": {"min_unique_eligible_per_country": gate_min, "per_country": gates},
        "domain_status_breakdown": {k: int(v) for k, v in
                                    rflags.loc[unique_mask, "domain_status"].value_counts().items()},
        "language_breakdown": {k: int(v) for k, v in
                               rflags.loc[dom_mask, "language"].value_counts().head(15).items()},
        "representativeness": {
            "platform_x_country": crosstab(elig, "site", "country_iso"),
            "country_x_quarter": crosstab(elig_q, "country_iso", "quarter"),
        },
    }

    report_path = OUT_ROOT / f"governance_ladder_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    flags_out = reps[["source_record_id", "batch_id", "stratum", "country_iso", "site"]].copy()
    for c in ["meta_valid", "date_in_window", "domain_status", "domain_ok", "content_ok",
              "near_exact_dup", "near_dup_minhash", "language", "language_allowed",
              "conditional_platform", "desc_signals"]:
        flags_out[c] = rflags[c]
    flags_out["eligible"] = elig_mask
    flags_out.astype(str).to_parquet(OUT_ROOT / f"record_flags_{stamp}.parquet", index=False)

    md = [f"# Governance ladder — {stamp} (config {rules['config_version']})", ""]
    md.append("| Denominator | N |")
    md.append("|---|---|")
    for k, v in report["ladder"].items():
        md.append(f"| {k} | {v:,} |")
    md.append("")
    md.append("## Eligible per stratum")
    for k, v in report["eligible_by_stratum"].items():
        md.append(f"- {k}: {v:,}")
    md.append("")
    md.append(f"## Coverage gate (>= {gate_min:,} unique eligible)")
    for c, g in gates.items():
        md.append(f"- {c}: {g['n_eligible']:,} {'PASS' if g['passes_gate'] else 'FAIL'}")
    (OUT_ROOT / f"governance_ladder_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
