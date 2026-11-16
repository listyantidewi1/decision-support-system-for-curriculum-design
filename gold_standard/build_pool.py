"""Build the sentence pool the gold-standard sample is drawn from.

Pipeline (all researcher-side, "catatan peneliti" per B-0 step 2-3):
  1. read the newest governance record_flags parquet -> eligible posting ids
     (+ their ladder language label and stratum/country/site);
  2. reload the raw batches, keep the first sighting of each posting (same as the
     ladder's representative set), and inner-join the eligible ids to recover the
     description text;
  3. cap postings per stratum (all of Indonesia kept, ASEAN/OECD down-sampled) so
     segmentation stays bounded while leaving ample headroom for pilot + main;
  4. segment each description into sentences with one sentence of context either
     side, and score each sentence (length band, relevance heuristic, Indonesian-
     token signal) to drive the stratified sample.

Run:  .venv-ml\\Scripts\\python.exe -m gold_standard.build_pool
Output: DATA/gold_standard/sentence_pool.parquet          (sentence rows)
        DATA/gold_standard/posting_shingles.parquet       (template signatures)

NOTE on the two provenance columns: `employer_hash` and `query_family` are
carried through from the raw batches because the group-aware split (Subbab
3.3.3) cannot enforce "postings from the same employer or template never
straddle train and test" without them. The pool on disk predates this change,
so a pool REBUILD is required before `gold_standard.split` can run on real
data; until then `gold_standard.posting_keys` reconstructs the same keys from
the raw batches as a sidecar, which leaves `sentence_pool.parquet` (and the
`pool_sha1_16` that `pilot_100_manifest.json` pins) byte-identical.
"""
from __future__ import annotations

import html as _htmllib
import json
import re
from pathlib import Path

import pandas as pd

from governance.ladder import REPO_ROOT, load_all_batches
from preprocess_jobs_pipeline import score_sentence_for_relevance_rescue, split_sentences
from . import SEED, SHINGLE_WORDS
from . import textsig
from .textsig import shingles_from_clean_text, signature_fingerprint

FLAGS_DIR = REPO_ROOT / "results" / "corpus_audits"
OUT_DIR = REPO_ROOT / "DATA" / "gold_standard"

# Keep all of Indonesia (small + boosted); down-sample the larger strata. Ample
# headroom for pilot 100 + main 1,200 + a challenge reserve at the sentence level.
POOL_POSTING_CAP = {"indonesia": None, "asean": 5000, "oecd": 8000}

_WS_RE = re.compile(r"[\s\u00a0]+")
# Several sources store the posting body as raw HTML (Kemnaker/Kitalulus/Kalibrr)
# or escaped markdown (Indeed/Glassdoor). Strip that presentation layer BEFORE
# sentence splitting so annotators label linguistic content, not markup, and span
# offsets index into clean text. Block-level tags become newlines (sentence/bullet
# boundaries that split_sentences understands); inline tags are dropped.
_BLOCK_TAG_RE = re.compile(
    r"(?i)<\s*/?\s*(p|div|li|ul|ol|tr|table|thead|tbody|h[1-6]|section|article|br)\b[^>]*>")
_TAG_RE = re.compile(r"<[^>]+>")
_MD_ESCAPE_RE = re.compile(r"\\([^\sA-Za-z0-9])")  # \-  \#  \&  ->  - # &
# Glyph bullets the splitter doesn't know (●○▪▸‣⁃◦ …); turn into a bullet form it
# does, so each list item becomes its own sentence.
_GLYPH_BULLET_RE = re.compile(r"[\u25CF\u25CB\u25AA\u25B8\u2023\u2043\u25E6\u25AB\u2219]")
# Line-start bullet markers (markdown '*'/ '\u2022'/'\u00B7') -> a boundary the splitter
# understands. Done BEFORE stripping '*', so list bullets survive while inline
# emphasis '*' and sentence-fusing '.*' residue are removed.
_MD_BULLET_LINE_RE = re.compile(r"(?m)^[ \t]*[\*\u2022\u00B7\+\-]\s+")
# Numbered / lettered list markers at line start ('1.', '2)', 'a.', 'iv)'), which
# no other rule handled: without this a numbered requirements list arrives as one
# fused sentence, or the marker attaches to the end of the previous sentence.
_NUM_BULLET_LINE_RE = re.compile(r"(?m)^[ \t]*(?:\d{1,2}|[a-zA-Z]|[ivxIVX]{1,4})[.)]\s+")


def _html_to_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Unescape BEFORE stripping tags, and twice: postings commonly arrive with
    # single- and double-escaped HTML (&lt;p&gt; / &amp;lt;p&amp;gt;). Unescaping
    # only after the tag pass left those entities in the text the annotators read
    # and index offsets into. Re-run the tag strip after unescaping so entities
    # that decode INTO tags are removed too.
    t = text
    for _ in range(2):
        t = _htmllib.unescape(t)
    t = _BLOCK_TAG_RE.sub("\n", t)
    t = _TAG_RE.sub(" ", t)
    for _ in range(2):                      # resolve single- and double-escaped md
        t = _MD_ESCAPE_RE.sub(r"\1", t)
    t = _MD_BULLET_LINE_RE.sub("\n\u2022 ", t)
    t = _NUM_BULLET_LINE_RE.sub("\n\u2022 ", t)
    t = _GLYPH_BULLET_RE.sub("\n\u2022 ", t)
    return t.replace("`", "").replace("*", "")  # strip emphasis + fused-sentence '*'
# A light Indonesian-token probe: common function words that rarely appear in
# English sentences. Two or more hits flags a sentence as Indonesian / mixed,
# giving a sentence-level bilingual signal even inside English-language postings.
_ID_TOKENS = {
    "dan", "atau", "yang", "dengan", "untuk", "pada", "dalam", "mampu", "serta",
    "akan", "dari", "ke", "di", "pengalaman", "kemampuan", "memiliki", "melakukan",
    "mengembangkan", "membuat", "menguasai", "tahun", "kerja", "mampu", "wajib",
    "minimal", "pengetahuan", "terhadap", "secara", "adalah", "tidak",
}


_LEADING_BULLET_RE = re.compile(r"^[\s\-\*•·●○▪◦#>]+")
_STRAY_STAR_RE = re.compile(r"\s\*+\s")


def _normalize_sentence(text: str) -> str:
    """Light normalization: strip residual bullet markers and collapse whitespace,
    but keep punctuation and casing so span offsets into the presented text remain
    faithful to the source the annotator sees (guideline Bagian 11)."""
    if not isinstance(text, str):
        return ""
    t = _WS_RE.sub(" ", text.replace("\r", " ").replace("\n", " "))
    t = _LEADING_BULLET_RE.sub("", t)        # drop leading bullet(s)
    t = _STRAY_STAR_RE.sub(" ", t)           # drop inline markdown bullet asterisks
    return _WS_RE.sub(" ", t).strip()


# Above this, a "sentence" is really a segmentation failure — a run-on posting with
# no terminal punctuation. Re-split it on secondary delimiters; any piece still over
# the cap is dropped from the sample downstream (not an annotatable unit).
MAX_SENT_CHARS = 300
# NOTE on delimiters: `\s{2,}` and `\t` were dead here — _normalize_sentence runs
# first and has already collapsed every whitespace run to a single space, so the
# run-on case this function exists for never saw them. The usable secondary
# delimiters on already-normalized text are terminal punctuation, bullet glyphs,
# and the conjunction/separator patterns typical of fused requirement lists.
_HARD_SPLIT_RE = re.compile(
    r"(?<=[.!?;:])\s+"            # terminal/clausal punctuation
    r"|\s*[•·●○▪◦]\s*"   # bullet glyphs
    r"|\s+[-–—]\s+"     # spaced hyphen / en- / em-dash separators
)


def _resplit_long(sent: str) -> list[str]:
    if len(sent) <= MAX_SENT_CHARS:
        return [sent]
    pieces = [p.strip() for p in _HARD_SPLIT_RE.split(sent) if p.strip()]
    return pieces or [sent]


def _length_band(n: int) -> str:
    if n < 60:
        return "short"
    if n <= 180:
        return "medium"
    return "long"


def _id_token_hits(text: str) -> int:
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    return sum(1 for t in toks if t in _ID_TOKENS)


def _latest_flags() -> Path:
    files = sorted(FLAGS_DIR.glob("record_flags_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no record_flags_*.parquet in {FLAGS_DIR}")
    return files[-1]


def build_pool() -> pd.DataFrame:
    flags_path = _latest_flags()
    print(f"eligible set <- {flags_path.name}", flush=True)
    flags = pd.read_parquet(
        flags_path,
        columns=["source_record_id", "stratum", "country_iso", "site",
                 "language", "eligible"],
    )
    elig = flags[flags["eligible"].astype(str) == "True"].copy()
    elig = elig.drop(columns=["eligible"])
    print(f"  eligible postings: {len(elig):,}", flush=True)

    print("loading raw batches (first sighting per posting) ...", flush=True)
    raw = load_all_batches()
    raw = (raw.sort_values(["acquisition_timestamp", "source_record_id"], kind="stable")
              .drop_duplicates("source_record_id"))   # deterministic first sighting
    # employer_hash + query_family are already in governance.ladder.LOAD_COLUMNS,
    # but the ladder's flags_out drops them and this projection used to drop them
    # too, so the split layer had no grouping key at all. They are carried here
    # (from the RAW batches, where add_provenance wrote them) because the
    # employer/template non-straddling rule (Subbab 3.3.3) is enforceable only
    # with them. query_family is the only occupation signal the corpus records.
    raw = raw[["source_record_id", "batch_id", "title", "description",
               "date_posted", "employer_hash", "query_family"]]

    df = elig.merge(raw, on="source_record_id", how="inner")
    print(f"  eligible postings with text: {len(df):,}", flush=True)

    # Cap postings per stratum (seeded) before segmentation.
    capped = []
    for stratum, grp in df.groupby("stratum", sort=False):
        cap = POOL_POSTING_CAP.get(stratum)
        capped.append(grp if cap is None or len(grp) <= cap
                      else grp.sample(n=cap, random_state=SEED))
    df = pd.concat(capped, ignore_index=True)
    print(f"  postings after stratum cap: {len(df):,}", flush=True)

    rows = []
    shingle_rows = []
    for rec in df.itertuples(index=False):
        clean = _html_to_text(rec.description)
        # Template signature, computed HERE because this is the only stage that
        # still holds the description text: the pool deliberately carries no
        # description (662k sentence rows x a posting body is unusable), so the
        # split layer would otherwise have to re-read all 25 raw batches. Written
        # to a sidecar, not into the pool, so the pool's row size is unchanged.
        sh = shingles_from_clean_text(clean)
        shingle_rows.append({
            "posting_id": rec.source_record_id,
            "n_shingles": int(sh.size),
            "shingles": sh.tolist(),
            "shingle_sha1_16": signature_fingerprint(sh),
        })
        sents = []
        for seg in split_sentences(clean):
            for piece in _resplit_long(_normalize_sentence(seg)):
                piece = _normalize_sentence(piece)   # re-strip markers on split pieces
                if len(piece) >= 15:        # drop degenerate fragments
                    sents.append(piece)
        n = len(sents)
        dt = pd.to_datetime(rec.date_posted, errors="coerce", utc=True)
        quarter = str(dt.to_period("Q")) if pd.notna(dt) else "NA"
        for i, s in enumerate(sents):
            score, rflags = score_sentence_for_relevance_rescue(s)
            rows.append({
                "sentence_id": f"{rec.source_record_id}#{i:03d}",
                "posting_id": rec.source_record_id,
                "batch_id": rec.batch_id,
                "employer_hash": rec.employer_hash if isinstance(rec.employer_hash, str) else "",
                "query_family": rec.query_family if isinstance(rec.query_family, str) else "",
                "stratum": rec.stratum,
                "country_iso": rec.country_iso,
                "site": rec.site,
                "posting_language": rec.language,
                "title": rec.title,
                "date_posted": rec.date_posted,
                "quarter": quarter,
                "sent_index": i,
                "n_sents_in_posting": n,
                "previous_sentence": sents[i - 1] if i > 0 else "",
                "sentence_text": s,
                "next_sentence": sents[i + 1] if i + 1 < n else "",
                "char_len": len(s),
                "length_band": _length_band(len(s)),
                "relevance_score": score,
                "relevance_flags": rflags,
                "id_token_hits": _id_token_hits(s),
            })
    pool = pd.DataFrame(rows)
    # Sentence-level language signal for the bilingual boost: Indonesian if the
    # posting is id OR the sentence itself trips the Indonesian-token probe.
    pool["is_indonesian_sentence"] = (
        (pool["posting_language"] == "id") | (pool["id_token_hits"] >= 2)
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sentence_pool.parquet"
    pool.to_parquet(out, index=False)

    shingles = pd.DataFrame(shingle_rows)
    sh_out = OUT_DIR / "posting_shingles.parquet"
    shingles.to_parquet(sh_out, index=False)
    # The signature PARAMETERS come from textsig itself. Four of them used to be
    # restated here, and `shingle_max_chars: 2500` had become a LIE the moment
    # the leakage window was removed: `shingles_from_clean_text(clean)` above is
    # called with no max_chars, so these signatures cover the FULL cleaned text.
    # A manifest that misdescribes the artifact beside it is worse than no
    # manifest, because the consumer trusts it.
    #
    # The `signature_cache` block is the FINGERPRINT `groups._try_sidecar`
    # demands before it will reuse this sidecar instead of re-shingling all 25
    # raw batches (~60 s). It is built from the `shingle_sha1_16` column that is
    # already computed above, so it costs nothing here.
    (OUT_DIR / "posting_shingles_manifest.json").write_text(json.dumps({
        "source": "gold_standard.build_pool",
        "pool_file": out.name,
        "flags_file": flags_path.name,
        "n_postings": int(len(shingles)),
        **textsig.signature_params(),
        "n_empty_signatures": int((shingles["n_shingles"] == 0).sum()),
        "signature_cache": textsig.cache_fingerprint(dict(zip(
            shingles["posting_id"].astype(str), shingles["shingle_sha1_16"]))),
    }, indent=2), encoding="utf-8")

    print(f"\nsentence pool: {len(pool):,} sentences from {df['source_record_id'].nunique():,} postings", flush=True)
    print("by stratum:\n", pool["stratum"].value_counts().to_string(), flush=True)
    print("indonesian-signal sentences:", int(pool["is_indonesian_sentence"].sum()), flush=True)
    print("employer_hash missing on", int((pool["employer_hash"] == "").sum()),
          "sentences (the employer rule is UNVERIFIABLE for those postings)", flush=True)
    print(f"-> {out}\n-> {sh_out}", flush=True)
    return pool


if __name__ == "__main__":
    build_pool()
