"""Gold-standard (data acuan) construction for Workstream B / tahap Design.

Implements the researcher-side ("catatan peneliti") steps of Paket B that precede
expert annotation: sentence segmentation of the eligible corpus, a stratified
sentence sample (pilot 100 -> main 1,200), and delivery of the sample to the three
annotators (AV/AI/AD) in two interchangeable formats — a Form A-1 workbook and a
Streamlit span annotator — both faithful to annotation guideline AG-RPL-LIMA-v3.0.

The guideline (B-1) and forms (B-2) themselves are validated by three experts with
Aiken's V (sheet V-B) before real annotation begins; this package only prepares and
delivers the data the annotators work on, which the guideline marks as researcher
work, not an instrument.
"""
from __future__ import annotations

GUIDELINE_CODE = "AG-RPL-LIMA-v3.0"
GOLD_STANDARD_VERSION = "0.1.0"

SEED = 42
PILOT_N = 100
MAIN_N = 1200
# Throwaway warm-up split, drawn disjoint from the pilot. Annotators practise the
# UI here so their first clicks never land in the gold standard (an interface
# trial once left 8 judged sentences in AV's pilot file).
PRACTICE_N = 10

# Five tags (Bagian 4). code -> full label.
TAGS = {
    "SH": "SKILL_HARD",
    "SS": "SKILL_SOFT",
    "KN": "KNOWLEDGE",
    "TT": "TOOL_TECHNOLOGY",
    "WB": "WORK_BEHAVIOR",
}

# Sentence-level relevance (Bagian 3.1) and the has_5tag marker (Bagian 3.2).
RELEVANCE = ["RELEVANT", "NON_RELEVANT", "UNCERTAIN"]
HAS_5TAG = ["YA", "TIDAK"]

# Annotator confidence (Bagian 8): 4 very sure .. 1 not sure.
CONFIDENCE = [4, 3, 2, 1]

# Special flags (Bagian 9). short code -> full name.
FLAGS = {
    "AL": "AMBIGUOUS_LABEL",
    "AB": "AMBIGUOUS_BOUNDARY",
    "IC": "INSUFFICIENT_CONTEXT",
    "OS": "OUT_OF_SCHEMA",
    "LU": "LANGUAGE_UNCERTAIN",
    "PD": "POSSIBLE_DUPLICATE",
    "GG": "GUIDELINE_GAP",
    "OP": "OVERLAP_POTENTIAL",
}

# Three expert annotators (B-0 §4): equal standing in initial annotation.
ANNOTATORS = {
    "AV": "Ahli Pendidikan Vokasi",
    "AI": "Ahli / praktisi industri RPL",
    "AD": "Ahli sains data / NLP",
}

# The three expert annotators read only English and Indonesian (including ID-EN
# code-mixing, guideline rule 7.13). The human-annotated gold standard is therefore
# restricted to these languages; non-EN/ID postings stay in the raw corpus and are
# handled downstream via the translation layer + the B-6 LLM-labeling audit, not by
# direct human annotation. This also drops the rare cross-border mis-stratified
# posting (e.g. a Vietnamese listing tagged Indonesia).
ANNOTATOR_LANGUAGES = ("en", "id")

# Pilot stratum quotas — Indonesian + bilingual boost (purposive, not corpus
# proportional): the dissertation targets Indonesian SMK RPL and a cross-language
# claim, so Indonesia is deliberately over-represented relative to its ~6% corpus
# share. Fractions sum to 1.0 and are applied to PILOT_N / MAIN_N.
STRATUM_QUOTA = {
    "indonesia": 0.45,
    "asean": 0.20,
    "oecd": 0.35,
}

# Within a stratum, aim for at least this share of Indonesian / mixed-language
# sentences where the stratum supplies them (exercises rules 7.13 / 7.14).
MIN_LOCAL_LANG_SHARE = {
    "indonesia": 0.60,
    "asean": 0.30,
    "oecd": 0.0,
}

# Deliberately keep a minority of low-relevance sentences so NON_RELEVANT is a real
# option the annotators exercise during calibration, rather than an empty category.
BORDERLINE_SHARE = 0.15

# ---------------------------------------------------------------------------
# Group-aware split layer (GROUPED SPLIT SPECIFICATION v1.0.0).
#
# The one MANDATORY clause of the proposal for this layer is a prohibition
# (Subbab 3.3.3 / PROP:1192, PROP:1436): "Lowongan yang berasal dari employer
# atau template yang sama tidak ditempatkan pada train dan test secara
# bersamaan" — postings from the same employer or the same (or very similar)
# template must never straddle train and test. Stratification of the split is
# explicitly best-effort ("sejauh memungkinkan"); the prohibition is not.
# Everything below is therefore organised so the prohibition is an assertion
# that runs BEFORE any byte is written, and the stratification is a reported
# target/achieved pair rather than a claim.
# ---------------------------------------------------------------------------
SPLIT_SPEC_VERSION = "1.0.0"

# -- group layer -------------------------------------------------------------
# Clustering threshold. Edges at a higher tau are a SUBSET of the edges at a
# lower tau, so a split that is leak-free at 0.30 is leak-free at 0.50/0.70 for
# free; the higher thresholds are therefore reported, never asserted (asserting
# a logically implied statement is not evidence).
TAU_BUILD = 0.30
TAU_ASSERT = (0.30, 0.50, 0.70)
# ABSOLUTE-OVERLAP FLOOR, in shared 5-word shingles. A ratio alone is not a
# template test: Jaccard is |A n B| / |A u B|, so a long posting DILUTES a large
# shared block and two ads can share an unbounded amount of verbatim text while
# staying under TAU_BUILD purely by being long. Measured on the real pool
# (18,667 postings, 626,082 sampleable sentences), the ratio-only rule left
# 16,522 posting pairs sharing TEN OR MORE byte-identical sampleable sentences
# in DIFFERENT leakage groups — free to straddle train/test.
#
# A pair is a template pair at `jaccard >= TAU_BUILD` OR `n_shared >= this`.
# 80 was chosen from a measured sweep, not guessed:
#
#   floor | edges   | groups | largest | %pool | leak>=10 | leak>=5
#   ratio |  13,430 | 11,764 |     251 | 1.34% |   16,522 |  37,181
#     100 |  98,396 |  9,833 |     277 | 1.48% |       24 |   1,534
#      80 | 105,349 |  9,644 |     277 | 1.48% |       10 |     998
#      70 | 110,567 |  9,502 |     283 | 1.52% |        8 |     724
#      60 | 114,844 |  9,355 |     429 | 2.30% |        6 |     556
#      30 | 162,127 |  8,472 |   2,886 |15.46% |        0 |      60
#
# 80 sits on the FLAT part of the cost curve (largest component identical to
# floor 100) while removing ~99.96% of the >=10-shared-sentence leak. Below 60
# the components cascade toward the ~20% share that makes an exact split
# infeasible. NOTE the sweep above used employer-EXACT edges only, so it
# UNDERSTATES the final grouping; the build also applies employer-loose edges.
#
# RESIDUAL, measured against the SHIPPED leakage_groups.parquet over a
# deliberately wider candidate net (J>=0.3 or n_shared>=5, 1,133,779 pairs) so
# that pairs the relation REJECTS are still counted:
#   of 50,115 pairs sharing >=10 byte-identical sampleable sentences, 6 straddle
#   of 95,558 pairs sharing  >=5 byte-identical sampleable sentences, 602 straddle
# This bound must stay published, never implied away. Floor 80 also brings the
# ASEAN stratum's largest group to 4.82% of 5,000 against a 5% guard (9 postings
# of headroom) — but that figure is IDENTICAL before the floor and at floor 100,
# so it is driven by employer grouping, not by this constant.
MIN_SHARED_SHINGLES_RESIDUAL = {
    "measured_against": "shipped leakage_groups.parquet",
    "candidate_net": "jaccard >= 0.30 or n_shared >= 5",
    "n_candidate_pairs": 1133779,
    "pairs_sharing_ge_10_sentences": 50115,
    "straddling_ge_10_sentences": 6,
    "pairs_sharing_ge_5_sentences": 95558,
    "straddling_ge_5_sentences": 602,
}
MIN_SHARED_SHINGLES = 80
# LSH recall band is built BELOW the clustering threshold so candidate recall
# near the cut is higher; every candidate is then verified by exact Jaccard.
LSH_THRESHOLD = 0.25
MINHASH_NUM_PERM = 128
SHINGLE_WORDS = 5
# LADDER-PARITY constant, NOT a group-layer one. The leakage layer compares the
# FULL cleaned text (`textsig.LEAKAGE_WINDOW_CHARS is None`): a 2,500-character
# window left 45% of the corpus's characters out of every template comparison
# and hid real cross-group clones, which was the first of the two fatal findings
# in the split review. This value survives only where parity with
# `governance.ladder.minhash_near_dups` is genuinely wanted, and is re-exported
# as `textsig.LADDER_PARITY_WINDOW_CHARS` to keep that intent visible at the use
# site. Do NOT feed it back into the leakage signature.
SHINGLE_MAX_CHARS = 2500

# -- sampling ----------------------------------------------------------------
# Main draw only (pilot/practice stay byte-identical). Caps every leakage group
# at 2 sentences, which (a) makes the parity proof in the allocator apply and
# (b) lowers within-split clustering, hence DEFF, hence the test CI.
MAX_SENT_PER_GROUP = 2

# -- allocation --------------------------------------------------------------
SPLIT_TARGETS = {"train": 720, "dev": 240, "test": 240}
AXIS_WEIGHTS = {"stratum": 8, "country_language": 4, "site": 2,
                "quarter": 2, "occupation_group": 2, "length_band": 1}
MAX_GROUP_SHARE = 0.25          # of the smallest part quota
MIN_CELL_SENTENCES = 10

# -- regimes -----------------------------------------------------------------
TEMPORAL_QUANTILE = 0.80        # most-recent 20% of each stratum -> temporal test
MIN_TEST_SENTENCES = 100

# -- leakage -----------------------------------------------------------------
SENT_NEAR_DUP_JACCARD = 0.90

# -- power / enlargement (CODE-ORIGIN, not proposal-origin) ------------------
# No numeric per-tag support threshold exists in the proposal or in Paket B v3,
# which assigns the computation to the researcher (B3:32). Derived instead from
# this repo's own claimability rule, gold_standard.iaa.CI_WIDTH_CLAIMABLE=0.20:
#   N >= 4 * 1.96^2 * 0.25 / 0.04 = 96.04 spans at worst-case p=0.5
#   x DEFF 1.18 = 113.3  ->  120
MIN_TEST_SUPPORT_SPANS_PER_TAG = 120
# Measured learning curve (this project, exact span match, distilbert, 8 epochs,
# 2 seeds): F1 .275@180 .354@360 .424@720 .468@1440 .472@2880; the 1440->2880
# gain (+0.0035) is inside the 0.030 seed spread, so one doubling of the train
# split is the largest defensible enlargement.
ENLARGE_MAX_GOLD_N = 2400
DESIGN_EFFECT = 1.18
ASSUMED_SPANS_PER_SENTENCE = 1.6   # assumption until pilot_100 is adjudicated

# Legal-form / geography tokens stripped by the LOOSE employer hash, which exists
# because employer_hash is an exact-string hash: "PT X" and "X Indonesia" hash
# differently, so exact-hash grouping is a LOWER BOUND on true employer leakage.
# The list is closed and committed, so the loose hash has no tuning knob.
LEGAL_FORM_TOKENS = frozenset("""
pt cv tbk persero pte ltd limited llc inc incorporated corp corporation co company
gmbh bv nv sa sarl plc ab oy as sdn bhd pty srl spa kk kft
group groups holding holdings ventures
international global worldwide
indonesia singapore malaysia philippines vietnam thailand asia apac emea
""".split())

# query_family -> occupation group. Near-synonymous scrape labels are merged;
# the two legacy scrape buckets are NOT guessed at (74% of OECD and 46% of ASEAN
# postings carry them), they resolve to UNRESOLVED. No keyword classifier.
OCCUPATION_MERGE_MAP = {
    "devops_cloud": "devops",
    "system_analyst": "system_analysis",
}
OCCUPATION_UNRESOLVED_FAMILIES = ("legacy_english_roles", "legacy_akkasi_roles")
OCCUPATION_UNRESOLVED = "UNRESOLVED"

# Cross-country hold-out eligibility rule (re-runnable by an examiner; the
# hold-out country is derived, never picked). The three annotators read only
# English and Indonesian, so a non-English hold-out cannot be annotated.
HOLDOUT_MIN_POOL_POSTINGS = 400
HOLDOUT_MIN_EMPLOYERS = 200
HOLDOUT_MIN_CANDIDATE_SENTENCES = 10000
HOLDOUT_CANDIDATE_CHAR_RANGE = (20, 300)
HOLDOUT_REQUIRED_LANGUAGE = "en"
PRIMARY_CLAIM_STRATUM = "indonesia"     # never held out

# Closed set of exclusion reasons (invariant I1 checks membership).
EXCLUSION_REASONS = (
    "pilot_group", "practice_group", "unverifiable_employer", "date_unparseable",
    "straddles_temporal_cut_dropped", "heldout_country_group",
)
