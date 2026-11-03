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
