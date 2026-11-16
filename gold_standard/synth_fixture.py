"""Synthetic fixture generator for the split layer.

The real `sentence_pool.parquet` predates the provenance columns the split layer
needs, and rebuilding it is a long segmentation run over the whole corpus. This
module generates a small, fully seeded, self-contained stand-in — raw batches,
a record_flags file and a sentence pool — so `posting_keys`, `groups`, `sample`
and `split` can be exercised end to end today, and so the tests have positive
controls for leakage rather than only negative ones.

The fixture deliberately PLANTS the structures the group layer exists to catch:
  * employers with several postings (exact-hash groups);
  * company-name variants that differ only by legal form ("PT Alpha" / "Alpha
    Ltd" / "Alpha Indonesia") so the LOOSE hash has something to merge;
  * near-identical descriptions (template clones) across different employers;
  * a bilingual Indonesian slice, a multi-country ASEAN/OECD slice and a date
    range wide enough for a temporal cut.

Nothing here touches the real DATA/ tree: every path is under the `root` given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_SENTENCE_FRAGMENTS = [
    "kandidat wajib memiliki pengalaman dalam pengembangan aplikasi web",
    "mampu bekerja dengan tim dan melakukan komunikasi yang baik",
    "menguasai pengetahuan dasar basis data dan pemrograman berorientasi objek",
    "memiliki kemampuan analisis terhadap kebutuhan sistem secara menyeluruh",
    "melakukan pengujian perangkat lunak dan dokumentasi hasil kerja",
]
EN_SENTENCE_FRAGMENTS = [
    "build and maintain scalable backend services using modern frameworks",
    "write clean testable code and participate in peer code review",
    "experience with relational databases and query optimisation is required",
    "collaborate with product owners to refine functional requirements",
    "familiarity with container orchestration and continuous delivery pipelines",
    "strong problem solving skills and attention to detail in production support",
]
# Large synthetic vocabulary: unrelated postings must NOT look like template
# clones, or the whole corpus collapses into one leakage group (the fixture's
# first version did exactly that, and groups.py correctly refused to write it).
VOCAB = [f"term{i:03d}" for i in range(400)] + [
    "kubernetes", "postgres", "typescript", "laravel", "golang", "pytest",
    "terraform", "graphql", "kafka", "redis", "flutter", "kotlin",
]
SITES_BY_STRATUM = {
    "indonesia": ["indeed", "jobstreet", "glints", "kitalulus"],
    "asean": ["indeed", "jobstreet"],
    "oecd": ["indeed", "linkedin"],
}
COUNTRIES = {
    "indonesia": ["ID"],
    "asean": ["SG", "MY", "PH", "VN", "TH"],
    "oecd": ["US", "GB", "CA", "AU", "DE"],
}
FAMILIES = ["software_developer", "backend_developer", "qa_engineer", "devops",
            "devops_cloud", "system_analysis", "system_analyst", "mobile_developer",
            "legacy_english_roles"]
LEGAL_FORMS = ["PT {n}", "{n} Ltd", "{n} Indonesia", "CV {n}", "{n}"]


def _descr(sentences: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{s}.</li>" for s in sentences) + "</ul>"


def build_fixture(root: Path, *, n_postings: int = 420, sents_per_posting: int = 5,
                  seed: int = 7, holdout_heavy: tuple[str, ...] = ("SG", "US"),
                  holdout_extra_sentences: int = 0, verbose: bool = False) -> dict:
    """Write a synthetic corpus under `root` and return the paths it created."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    raw_dir = root / "DATA" / "corpus_raw" / "batch-synth-000"
    gs_dir = root / "DATA" / "gold_standard"
    flags_dir = root / "results" / "corpus_audits"
    for d in (raw_dir, gs_dir, flags_dir):
        d.mkdir(parents=True, exist_ok=True)

    # stratum mix mirrors STRATUM_QUOTA's purposive Indonesian boost so a draw of
    # 45/20/35 is actually satisfiable from the fixture
    strata = (["indonesia"] * int(n_postings * 0.45)
              + ["asean"] * int(n_postings * 0.25)
              + ["oecd"] * (n_postings - int(n_postings * 0.45) - int(n_postings * 0.25)))

    from acquisition.schema import make_employer_hash

    raw_rows, pool_rows = [], []
    languages: dict[str, str] = {}
    prev_bodies: list[str] = []
    base_companies = [f"Nusantara {i}" for i in range(n_postings)]
    for idx, stratum in enumerate(strata):
        pid = f"u-synth{idx:06d}"
        country = COUNTRIES[stratum][idx % len(COUNTRIES[stratum])]
        if idx % 7 == 0 and stratum == "asean":
            country = holdout_heavy[0]
        if idx % 7 == 0 and stratum == "oecd":
            country = holdout_heavy[1]
        site = SITES_BY_STRATUM[stratum][idx % len(SITES_BY_STRATUM[stratum])]
        family = FAMILIES[idx % len(FAMILIES)]

        # --- employers: ~15% repeat an earlier employer exactly (exact-hash
        #     groups), ~8% reuse an earlier NAME in a different legal form (loose
        #     groups only), ~4% have no company at all (unverifiable employer).
        name_idx = idx
        if idx % 7 == 3 and idx >= 7:
            name_idx = idx - 7                      # exact repeat
        form = LEGAL_FORMS[-1]
        if idx % 13 == 5 and idx >= 13:
            name_idx, form = idx - 13, LEGAL_FORMS[idx % len(LEGAL_FORMS)]
        company = form.format(n=base_companies[name_idx])
        if idx % 25 == 11:
            company = ""                            # company-less posting

        language = "id" if (stratum == "indonesia" and idx % 2 == 0) else "en"
        frags = ID_SENTENCE_FRAGMENTS if language == "id" else EN_SENTENCE_FRAGMENTS

        # --- template clones: every 11th posting copies the PREVIOUS posting's
        #     sentence bodies (different employer, near-identical text) so the
        #     MinHash/Jaccard layer has real template pairs to find.
        clone = (idx % 11 == 4 and idx > 0 and prev_bodies)
        n_sents = sents_per_posting + (
            holdout_extra_sentences if country in holdout_heavy else 0)
        bodies, sentences = [], []
        for s in range(n_sents):
            if clone:
                body = prev_bodies[s % len(prev_bodies)]
            else:
                # A realistic opener plus a posting-specific word sequence. The
                # opener is shared, so unrelated postings share a handful of
                # shingles (Jaccard well under TAU_BUILD) while a clone shares all
                # of them — which is the contrast the group layer has to resolve.
                words = [VOCAB[int(v)] for v in rng.integers(0, len(VOCAB), 11)]
                body = frags[(idx + s) % len(frags)] + " " + " ".join(words)
            bodies.append(body)
            # the per-sentence id keeps the TEXT unique (sample.py dedups exact
            # sentence text pool-wide) while leaving the 5-word shingles shared,
            # so a clone is a template pair and not a duplicate row
            sentences.append(f"{body} [{pid}-{s:02d}]")
        prev_bodies = bodies
        languages[pid] = language

        day = 1 + int(rng.integers(0, 330))
        date = (pd.Timestamp("2025-12-01", tz="UTC") + pd.Timedelta(days=day))
        date_str = (date.strftime("%Y-%m-%dT%H:%M:%SZ") if site != "indeed"
                    else date.strftime("%Y-%m-%d"))
        if idx == 3:
            date_str = "not-a-date"                 # exercises the quarantine path

        raw_rows.append({
            "source_record_id": pid,
            "acquisition_timestamp": f"2026-11-0{1 + idx % 9}T00:00:00Z",
            "company": company,
            "employer_hash": make_employer_hash(company),
            "query_family": family,
            "date_posted": date_str,
            "description": _descr(sentences),
            "title": f"{family.replace('_', ' ').title()} {idx}",
            "site": site, "stratum": stratum, "country_iso": country,
            "batch_id": "batch-synth-000",
        })
        for s, text in enumerate(sentences):
            pool_rows.append({
                "sentence_id": f"{pid}#{s:03d}", "posting_id": pid,
                "batch_id": "batch-synth-000",
                "employer_hash": make_employer_hash(company),
                "query_family": family,
                "stratum": stratum, "country_iso": country, "site": site,
                "posting_language": language,
                "title": f"{family.replace('_', ' ').title()} {idx}",
                "date_posted": date_str,
                "quarter": str(date.tz_localize(None).to_period("Q")),
                "sent_index": s, "n_sents_in_posting": n_sents,
                "previous_sentence": sentences[s - 1] if s else "",
                "sentence_text": text,
                "next_sentence": sentences[s + 1] if s + 1 < n_sents else "",
                "char_len": len(text),
                "length_band": "short" if len(text) < 60 else (
                    "medium" if len(text) <= 180 else "long"),
                # keep a real low-relevance minority so BORDERLINE_SHARE is fillable
                "relevance_score": 1.0 if s == n_sents - 1 else 3.0,
                "relevance_flags": "",
                "id_token_hits": 3 if language == "id" else 0,
                "is_indonesian_sentence": language == "id",
            })

    raw = pd.DataFrame(raw_rows)
    raw.to_parquet(raw_dir / "records.parquet", index=False)

    flags = raw[["source_record_id", "batch_id", "stratum", "country_iso", "site"]].copy()
    flags["language"] = flags["source_record_id"].map(languages)
    flags["eligible"] = "True"
    flags_path = flags_dir / "record_flags_20261116T000000Z.parquet"
    flags.to_parquet(flags_path, index=False)

    pool = pd.DataFrame(pool_rows)
    pool_path = gs_dir / "sentence_pool.parquet"
    pool.to_parquet(pool_path, index=False)

    meta = {
        "root": str(root), "seed": seed,
        "n_postings": int(len(raw)), "n_pool_sentences": int(len(pool)),
        "raw_batch": str(raw_dir / "records.parquet"),
        "flags": str(flags_path), "pool": str(pool_path),
        "by_stratum": pool["stratum"].value_counts().to_dict(),
        "by_country": pool["country_iso"].value_counts().to_dict(),
        "n_company_less_postings": int((raw["employer_hash"] == "").sum()),
        "n_unparseable_dates": int((raw["date_posted"] == "not-a-date").sum()),
    }
    (gs_dir / "synth_fixture_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(meta, indent=2), flush=True)
    return meta


def write_registry(path: Path, targets: dict[str, int], name: str = "CONFIG_SMALL",
                   ) -> Path:
    """A throwaway, correctly-hashed registry so tests can drive the registry gate
    with fixture-sized targets instead of the pre-registered 720/240/240."""
    import hashlib

    body = {"primary": True, "runnable": True, "targets": dict(targets),
            "origin": "TEST FIXTURE ONLY — not a pre-registered configuration"}
    entry = {
        "registered_utc": "2026-11-16T00:00:00+00:00",
        "registered_git_commit": "fixture",
        "frozen": True,
        "content_sha256": hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")).hexdigest(),
        "body": body,
    }
    Path(path).write_text(json.dumps(
        {"registry_version": "fixture", "configs": {name: entry}}, indent=2),
        encoding="utf-8")
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-postings", type=int, default=420)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    build_fixture(Path(args.root), n_postings=args.n_postings, seed=args.seed,
                  verbose=True)


if __name__ == "__main__":
    main()
