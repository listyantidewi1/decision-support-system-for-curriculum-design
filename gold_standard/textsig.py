"""Deterministic template signatures (shingle sets) for posting descriptions.

This module is deliberately PURE and import-light: `build_pool` computes the
signature while the description text is still in scope (it is the only stage
that holds it), and `groups` recomputes it from the raw batches when the pool
sidecar is absent or cannot be verified. Both must produce byte-identical
signatures, so the hashing lives here once.

THE WINDOW (finding F01, FATAL). The signature used to be taken over the first
`SHINGLE_MAX_CHARS = 2500` characters of the cleaned description, a value
inherited from `governance/configs/tabel_3_9_v1.json`'s `duplication.near_dup`
block — where it applies to RAW text for a different purpose (0.85 exact-dup
detection inside one batch). Carrying it into the leakage layer made the
PROP:1192/PROP:1436 guarantee cover a prefix rather than a posting: 69.0% of
the 18,667 pool postings are longer than 2,500 cleaned characters (median
3,705, q0.90 7,004, q0.99 10,953, max 26,225), and on the real corpus 126
posting pairs whose FULL cleaned-text Jaccard is >= TAU_BUILD sat in different
leakage groups purely because of the cut — including two ITALDESIGN postings at
full Jaccard 0.7374 whose truncated Jaccard is 0.2814.

The window is therefore REMOVED, not raised: `LEAKAGE_WINDOW_CHARS = None`
means the whole cleaned description is shingled. Measured on all 18,667 pool
postings (`.venv-ml` CPU, this machine, no profiler attached):

    stage                      2,500-char window   full cleaned text
    raw description read            13.5 s              13.5 s   (shared)
    _html_to_text                    7.2 s               7.2 s   (shared)
    shingling                       24.8 s              44.5 s   (+19.7 s)
    exact pair enumeration           4.3 s               8.2 s   (+3.9 s)
    signature set in memory         42.7 MB             78.0 MB  (+35.3 MB)
    mean shingles per posting        299.9               547.7
    pairs found at Jaccard >= 0.30   5,319              13,430

so the whole signature + edge stage costs 49.8 s truncated against 73.4 s at
full text, and a complete `gold_standard.groups` run over the real pool takes
about 95 s wall clock. A raised-but-finite window was rejected because q0.99 is
already 10,953 characters, so any finite value still truncates a tail of real
postings while re-introducing exactly the free parameter that caused the
defect; "the whole posting" has no knob.

`SHINGLE_MAX_CHARS` is kept, exported and unchanged as
`LADDER_PARITY_WINDOW_CHARS`, for the one place where parity with
`governance.ladder.minhash_near_dups` is actually required. It is no longer the
leakage window.

A window is only safe if it cannot be silently swapped, so every signature
cache carries a FINGERPRINT (`cache_fingerprint`) over the window, the cleaner,
the shingle width, the hash function, the cache's own posting-id set and its
per-posting signature digests. A consumer that cannot match the fingerprint
must REFUSE the cache (`gold_standard.groups.load_shingles`), never accept it
on posting-id coverage — finding F05/F14, where `build_pool`'s sidecar was
trusted on coverage alone and `shingle_sha1_16` was written for this check and
read by nobody.

Two deviations from `governance.ladder.minhash_near_dups` remain deliberate:

  * the signature is taken over CLEANED text (`build_pool._html_to_text`), not
    raw HTML, so the same template posted through two sites with different
    wrappers matches. Consequence that must be reported, not hidden: the
    ladder's 0.85 dedup ran on RAW text, so cleaned-text pairs at >= 0.85 can
    still exist inside the eligible set even though 0 eligible rows carry
    `near_dup_minhash` (170 such pairs on the real pool);
  * shingles are hashed with blake2b-64, never python's `hash()`, whose
    PYTHONHASHSEED randomisation would destroy determinism across processes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import MINHASH_NUM_PERM, SHINGLE_MAX_CHARS, SHINGLE_WORDS

# Bumped from an implicit v1 when the leakage window changed: a v1 cache and a
# v2 cache are not interchangeable, and the fingerprint says so out loud.
SIGNATURE_SPEC = "leakage-signature/2"

HASH_FN = "blake2b-64-masked-63"
# Parquet has no unsigned 64-bit integer in its default pandas round-trip, so a
# full 2**64 hash overflows on write. The top bit is masked off: 63 bits of
# entropy, every value representable as int64, and the mask is part of the hash
# definition so signatures stay byte-identical across writers and processes.
_MASK63 = (1 << 63) - 1
_MASK_BITS = 63

# The leakage window. None == the entire cleaned description (see the module
# docstring for the measurement that chose it over any finite value).
LEAKAGE_WINDOW_CHARS: int | None = None
# The ladder's raw-text near-duplicate window, kept only where parity with
# governance.ladder.minhash_near_dups is required. NOT the leakage window.
LADDER_PARITY_WINDOW_CHARS: int = SHINGLE_MAX_CHARS


def window_label(window_chars: int | None = LEAKAGE_WINDOW_CHARS) -> str:
    """Human-readable name of a window, used in manifests and error messages."""
    if window_chars is None:
        return "full cleaned text (no character window)"
    return f"first {int(window_chars)} cleaned characters"


def text_pipeline(window_chars: int | None = LEAKAGE_WINDOW_CHARS,
                  shingle_words: int = SHINGLE_WORDS) -> str:
    return (f"build_pool._html_to_text, {window_label(window_chars)}, "
            f"lowercased, whitespace-split, {int(shingle_words)}-word shingles, "
            f"{HASH_FN}")


TEXT_PIPELINE = text_pipeline()


def _shingle_u64(shingle: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
        "big") & _MASK63


def shingles_from_clean_text(clean_text: str,
                            shingle_words: int = SHINGLE_WORDS,
                            max_chars: int | None = LEAKAGE_WINDOW_CHARS) -> np.ndarray:
    """Sorted unique uint64 shingle hashes of already-cleaned description text.

    `max_chars=None` (the default, and the only value the leakage layer uses)
    shingles the WHOLE cleaned description. A finite `max_chars` truncates
    BEFORE the word split, matching `governance.ladder.minhash_near_dups`, and
    exists only for explicit ladder-parity callers.
    """
    if not isinstance(clean_text, str) or not clean_text.strip():
        return np.zeros(0, dtype=np.uint64)
    window = clean_text if max_chars is None else clean_text[:int(max_chars)]
    words = window.lower().split()
    k = shingle_words
    n = max(len(words) - k + 1, 1)
    shingles = {" ".join(words[i:i + k]) for i in range(n)}
    return np.array(sorted({_shingle_u64(s) for s in shingles}), dtype=np.uint64)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard over two sorted unique uint64 shingle arrays."""
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return 0.0 if union == 0 else inter / union


def overlap_and_jaccard(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    """Both leakage quantities in ONE intersection pass.

    The template relation is "jaccard >= tau OR n_shared >= floor", so every
    caller needs both numbers. Computing them with two separate intersect1d
    calls doubled the cost of the all-pairs invariant (I7 compares ~400k
    cross-part pairs), so they are returned together.
    """
    if a.size == 0 or b.size == 0:
        return 0, 0.0
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    union = a.size + b.size - inter
    return inter, (0.0 if union == 0 else inter / union)


def minhash(shingles: np.ndarray, num_perm: int = MINHASH_NUM_PERM):
    """datasketch MinHash over an already-hashed shingle array.

    Each uint64 is fed as its 8 big-endian bytes, so the MinHash depends only on
    the shingle set, not on insertion order or on the text pipeline running twice.

    NOTE: as of finding F17 this is no longer used to generate template
    candidates. `gold_standard.groups.template_pairs` now enumerates every pair
    sharing at least one shingle exactly (sparse inverted index), which is both
    complete at every tau and ~20x faster than MinHashLSH on this corpus. The
    helper is kept because MinHash is still the right tool for an approximate
    pass over a corpus too large for the exact index.
    """
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    for v in shingles.tolist():
        mh.update(int(v).to_bytes(8, "big"))
    return mh


def signature_fingerprint(shingles: np.ndarray) -> str:
    """Short stable digest of ONE posting's shingle set."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(shingles, dtype=">u8").tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------- cleaner probe
# A signature depends on the CLEANER as much as on the window, and the cleaner
# (`build_pool._html_to_text`) is a pile of regexes that may be edited. Rather
# than a hand-maintained version string that nobody remembers to bump, the
# cleaner is fingerprinted BEHAVIOURALLY: these committed probes exercise every
# rule it has (double-escaped entities, block and inline tags, markdown escapes,
# markdown / numbered / glyph bullets, backticks and emphasis, whitespace), and
# the fingerprint is a digest of its outputs. Any behaviour change that could
# move a shingle boundary changes the digest and therefore invalidates every
# cache built under the old cleaner.
CLEANER_PROBES = (
    "&lt;p&gt;Requirements:&lt;/p&gt;&lt;ul&gt;&lt;li&gt;3 years Java&lt;/li&gt;&lt;/ul&gt;",
    "&amp;lt;div&amp;gt;Tanggung jawab&amp;lt;/div&amp;gt;\n* Membangun REST API\n* Menulis unit test",
    "<p>Skills</p><br/>1. Python &amp; SQL\n2) Docker\niv) Kubernetes</p>",
    "● Agile\n▪ Scrum\n• CI/CD `pipelines` and *ownership*",
    "Tabs\tand   multiple   spaces plus \\*escaped\\* markdown.",
    "",
)
_CLEANER_FINGERPRINT: str | None = None


def cleaner_fingerprint() -> str:
    """Behavioural digest of `build_pool._html_to_text` over `CLEANER_PROBES`."""
    global _CLEANER_FINGERPRINT
    if _CLEANER_FINGERPRINT is None:
        from .build_pool import _html_to_text          # lazy: build_pool imports us
        h = hashlib.sha256()
        for probe in CLEANER_PROBES:
            h.update(_html_to_text(probe).encode("utf-8"))
            h.update(b"\x00")
        _CLEANER_FINGERPRINT = h.hexdigest()[:16]
    return _CLEANER_FINGERPRINT


# ------------------------------------------------------------- cache fingerprint
def signature_params(window_chars: int | None = LEAKAGE_WINDOW_CHARS,
                     shingle_words: int = SHINGLE_WORDS) -> dict:
    """Every parameter a signature depends on. Recorded, and fingerprinted."""
    return {
        "signature_spec": SIGNATURE_SPEC,
        "window_chars": None if window_chars is None else int(window_chars),
        "window": window_label(window_chars),
        "shingle_words": int(shingle_words),
        "hash_fn": HASH_FN,
        "mask_bits": _MASK_BITS,
        "cleaner": "gold_standard.build_pool._html_to_text",
        "cleaner_probe_sha256_16": cleaner_fingerprint(),
        "text_pipeline": text_pipeline(window_chars, shingle_words),
    }


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def params_fingerprint(params: dict | None = None, **kwargs) -> str:
    """Digest of the signature PARAMETERS alone (window, cleaner, width, hash)."""
    p = params if params is not None else signature_params(**kwargs)
    return hashlib.sha256(_canon(p)).hexdigest()[:16]


def pool_identity(posting_ids) -> str:
    """Digest of the posting-id SET a cache claims to describe."""
    h = hashlib.sha1()
    for pid in sorted(map(str, posting_ids)):
        h.update(pid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def cache_content_fingerprint(shingles) -> str:
    """Full sha256 over '<posting_id>\\t<signature_fingerprint>\\n' lines.

    Byte-identical to `gold_standard.split_assert.shingle_cache_fingerprint`, so
    the value the group layer RECORDS is the value I7 RECOMPUTES from the cache
    it is handed; `tests/test_split_groups.py` asserts the two agree. Values may
    be either shingle arrays or already-computed per-posting fingerprints.
    """
    h = hashlib.sha256()
    for pid in sorted(map(str, shingles)):
        v = shingles[pid]
        fp = v if isinstance(v, str) else signature_fingerprint(
            np.asarray(v, dtype=np.uint64))
        h.update(f"{pid}\t{fp}\n".encode("utf-8"))
    return h.hexdigest()


def cache_fingerprint(shingles, *,
                      window_chars: int | None = LEAKAGE_WINDOW_CHARS,
                      shingle_words: int = SHINGLE_WORDS) -> dict:
    """The full fingerprint block a signature cache must carry.

    Covers all four ways a cache can be wrong: wrong PARAMETERS (a cache built
    under a 2,500-char window reused under the full-text window, or under an
    edited cleaner), wrong POSTINGS (rows added or removed — the attack that
    turned I7 into `n_pairs_compared: 0`), wrong CONTENT (right posting ids,
    garbage signatures — the attack that silently produced 0 template edges),
    and a hand-edited manifest (the block digests itself).
    """
    params = signature_params(window_chars, shingle_words)
    ids = sorted(map(str, shingles))
    body = {
        "signature_spec": SIGNATURE_SPEC,
        "signature_params": params,
        "signature_params_fingerprint": params_fingerprint(params),
        "n_postings": len(ids),
        "pool_identity_sha1_16": pool_identity(ids),
        "content_sha256": cache_content_fingerprint(shingles),
        "content_fingerprint_algorithm": (
            "sha256 over '<posting_id>\\t<textsig.signature_fingerprint("
            "signature)>\\n' lines, posting ids sorted ascending"),
    }
    body["cache_fingerprint"] = hashlib.sha256(_canon(body)).hexdigest()[:16]
    return body


def verify_cache_fingerprint(declared: dict | None, shingles, *,
                             window_chars: int | None = LEAKAGE_WINDOW_CHARS,
                             shingle_words: int = SHINGLE_WORDS,
                             require_ids=None) -> list[str]:
    """Reasons the declared fingerprint does not describe `shingles`; [] == OK.

    Fail-closed by construction: a MISSING or unparseable declaration is a
    mismatch, not a pass. Posting-id coverage is checked SEPARATELY and is never
    sufficient on its own — accepting a cache on coverage alone is finding
    F05/F14.
    """
    problems: list[str] = []
    if not isinstance(declared, dict) or not declared.get("cache_fingerprint"):
        problems.append(
            "no cache fingerprint is recorded, so the window, the cleaner, the "
            "shingle width and the signature contents are all unverifiable; a "
            "signature cache is only usable with a fingerprint")
        return problems

    # (1) the declaration must digest itself: catches a hand-edited manifest.
    body = {k: v for k, v in declared.items() if k != "cache_fingerprint"}
    allowed = {"signature_spec", "signature_params", "signature_params_fingerprint",
               "n_postings", "pool_identity_sha1_16", "content_sha256",
               "content_fingerprint_algorithm"}
    body = {k: v for k, v in body.items() if k in allowed}
    if hashlib.sha256(_canon(body)).hexdigest()[:16] != declared["cache_fingerprint"]:
        problems.append(
            "the declared cache_fingerprint does not digest its own fields: the "
            "manifest has been edited by hand or written by an incompatible "
            "producer")

    actual = cache_fingerprint(shingles, window_chars=window_chars,
                               shingle_words=shingle_words)

    # (2) parameters, field by field, so the message NAMES what changed.
    dp = declared.get("signature_params") or {}
    ap = actual["signature_params"]
    if not isinstance(dp, dict):
        problems.append("signature_params is not an object")
    else:
        for key in sorted(ap):
            if key not in dp:
                problems.append(f"signature_params.{key} is not recorded")
            elif dp[key] != ap[key]:
                problems.append(
                    f"signature_params.{key}: cache says {dp[key]!r}, this code "
                    f"requires {ap[key]!r}")
    if declared.get("signature_params_fingerprint") != actual["signature_params_fingerprint"]:
        problems.append(
            f"signature parameters differ: cache "
            f"{declared.get('signature_params_fingerprint')!r} vs code "
            f"{actual['signature_params_fingerprint']!r}")

    # (3) the id set and (4) the contents.
    if declared.get("pool_identity_sha1_16") != actual["pool_identity_sha1_16"]:
        problems.append(
            f"the cache describes a different posting set: declared "
            f"{declared.get('pool_identity_sha1_16')!r} vs the "
            f"{actual['n_postings']} rows present "
            f"({actual['pool_identity_sha1_16']!r})")
    if declared.get("n_postings") != actual["n_postings"]:
        problems.append(f"n_postings: declared {declared.get('n_postings')!r}, "
                        f"present {actual['n_postings']}")
    if str(declared.get("content_sha256", "")).lower() != actual["content_sha256"]:
        problems.append(
            "the signatures themselves differ from the ones the fingerprint was "
            "written over (right posting ids, changed contents)")

    if require_ids is not None:
        missing = set(map(str, require_ids)) - set(map(str, shingles))
        if missing:
            problems.append(
                f"{len(missing)} required postings have no signature in the "
                f"cache; first few: {sorted(missing)[:5]}")
    return problems
