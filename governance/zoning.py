"""Which sentences of a posting state a duty or a requirement, and which do not.

A job posting is not only requirements. It also describes the company, lists
benefits, states its equal-opportunity policy and tells the reader how to
apply. The rule extractor tags any phrase that opens with an action verb, so
"support you and your family", "build on our values of respect" and "apply
early" became hard skills, then groups, then "valid" competencies. On
2026-11-20 a random sample of 24 valid families held about 10 of that kind
(experiment log, EXP-030). This module removes such sentences before the
competency layers count anything.

How, in three steps:

  1. HEADERS. Many postings carry section headers as short sentences of their
     own ("Responsibilities", "Qualifications", "What we offer", "Tentang
     kami"). A lexicon recognises them and gives each a zone:
         duty, requirement         -> a sentence under it is KEPT
         company, benefit, admin   -> a sentence under it is DROPPED
     Every sentence takes the zone of the last header before it in the same
     posting. These are WEAK labels: nobody checked them, and a header can be
     wrong about a sentence below it.
  2. A CLASSIFIER learns from those weak labels (word and two-word features,
     logistic regression) to tell kept from dropped sentences by their wording.
     It is for the sentences no header covers: postings without headers, and
     the text before a posting's first header.
  3. DECISION. Under a header the header decides. Elsewhere the classifier
     decides, and drops a sentence only when it is confident
     (`drop_below`): losing a real requirement costs more than letting one
     line of boilerplate through.

What this is not. It is not validated against people. The agreement figure it
reports is agreement with its own weak labels on postings it did not train on,
which says the classifier learned the headers' distinction, not that the
distinction is right. The gold standard, when it exists, will show how many
expert-tagged spans sit in sentences this filter drops; that is the number to
report. Until then it is an engineering filter and can be switched off
(`--no-zone-filter`) to see what it changes.

Reads the sentence pool; writes only under `DATA/competency/`.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

ZONING_VERSION = "zoning-v1"
KEEP_ZONES = ("duty", "requirement")
DROP_ZONES = ("company", "benefit", "admin")

# A header is a short sentence that IS one of these, give or take a qualifier.
_HEADERS: dict[str, str] = {
    "duty": (r"(?:key |main |core |primary |job |your |day[- ]to[- ]day |essential )?"
             r"(?:responsibilit(?:y|ies)|duties(?: and responsibilities)?|job descriptions?|"
             r"description|role description|what (?:you(?:'ll| will)|will you) (?:do|be doing)\??|"
             r"what you(?:'ll| will) be working on|in this role,? you will|the role|about the role|"
             r"the opportunity|position (?:overview|summary)|job summary|role overview|"
             r"deskripsi pekerjaan|deskripsi|tanggung jawab(?: utama| pekerjaan)?|tugas(?: utama)?|"
             r"tugas dan tanggung jawab|uraian pekerjaan)"),
    "requirement": (r"(?:minimum |basic |preferred |required |desired |essential |job |key |"
                    r"technical |additional |general )?"
                    r"(?:requirements?|qualifications?|skills(?: (?:and|&) experience)?|"
                    r"skills required|experience|competenc(?:y|ies)|"
                    r"what (?:we(?:'re| are)|we) look(?:ing)? for|who you are|"
                    r"what you(?:'ll| will) bring|what you(?:'ll| will) need|"
                    r"your skills and experience|must[- ]haves?|nice[- ]to[- ]haves?|"
                    r"about you|you have|kualifikasi|persyaratan(?: pekerjaan)?|"
                    r"syarat(?: pekerjaan)?|keahlian|kemampuan)"),
    "benefit": (r"(?:benefits?(?: (?:and|&) perks)?|perks(?: (?:and|&) benefits)?|what we offer|"
                r"we offer|what(?:'s| is) in it for you\??|why (?:join|work with) us\??|"
                r"why you(?:'ll| will) love (?:working here|it here|us)|"
                r"what you(?:'ll| will) get|compensation(?: (?:and|&) benefits)?|"
                r"salary(?: range)?|pay range|our offer|rewards|"
                r"fasilitas|benefit|keuntungan|tunjangan|yang kami tawarkan)"),
    "company": (r"(?:company (?:description|overview|profile)|about (?:us|the company|the team|"
                r"our team|the business|the department)|who we are|our (?:story|mission|values|"
                r"culture|company)|life at .{1,30}|tentang (?:kami|perusahaan)|profil perusahaan)"),
    "admin": (r"(?:additional information|application questions?(?:\(s\))?|how to apply|"
              r"application process|hiring process|interview process|equal opportunity(?: employer)?|"
              r"eeo(?: statement)?|diversity(?: (?:and|&) inclusion)?|accommodations?|"
              r"job types?:.*|work location:.*|employment type:?.*|schedule:.*|pay:.*|"
              r"work authori[sz]ation|disclaimer|privacy notice|cara melamar|informasi tambahan)"),
}
_HEADER_RE = {zone: re.compile(rf"^{pattern}$", re.IGNORECASE) for zone, pattern in _HEADERS.items()}
_EDGE = " \t:;-–—*•·#>|.!\"'"
MAX_HEADER_WORDS = 8


@dataclass(frozen=True)
class ZoningConfig:
    drop_below: float = 0.5           # no header above: drop only if P(keep) is under this
    min_df: int = 5
    max_features: int = 200_000
    test_share: float = 0.2           # postings held out to report agreement with the weak labels
    seed: int = 20261120
    version: str = ZONING_VERSION


def header_zone(text: str) -> str | None:
    """The zone a sentence announces, or None when it is not a header."""
    t = text.replace("’", "'").strip(_EDGE)
    if not t or len(t.split()) > MAX_HEADER_WORDS:
        return None
    for zone, rx in _HEADER_RE.items():
        if rx.match(t):
            return zone
    return None


def weak_labels(posting_ids, sent_indexes, texts) -> list[str | None]:
    """For every sentence: 'header' if it is one, else the zone of the last
    header before it in its posting, else None. Input order does not matter."""
    order = sorted(range(len(texts)), key=lambda i: (posting_ids[i], sent_indexes[i]))
    out: list[str | None] = [None] * len(texts)
    current_posting, zone = None, None
    for i in order:
        if posting_ids[i] != current_posting:
            current_posting, zone = posting_ids[i], None
        z = header_zone(texts[i])
        if z:
            zone, out[i] = z, "header"
        else:
            out[i] = zone
    return out


def fit(posting_ids, sent_indexes, texts, cfg: ZoningConfig | None = None) -> dict:
    """Weak labels, classifier, decision for every sentence.

    Returns {"keep": list[bool], "weak": list, "p_keep": list[float], "report": dict}."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    cfg = cfg or ZoningConfig()
    weak = weak_labels(posting_ids, sent_indexes, texts)
    y = np.array([1 if w in KEEP_ZONES else 0 if w in DROP_ZONES else -1 for w in weak])
    labelled = np.where(y >= 0)[0]
    report: dict = {"zoning_version": cfg.version, "settings": asdict(cfg),
                    "n_sentences": len(texts),
                    "n_headers": int(sum(1 for w in weak if w == "header")),
                    "n_under_a_header": int(len(labelled)),
                    "weak_zone_counts": {z: int(sum(1 for w in weak if w == z))
                                         for z in KEEP_ZONES + DROP_ZONES},
                    "n_postings_with_a_header": len({posting_ids[i] for i, w in enumerate(weak)
                                                     if w == "header"}),
                    "n_postings": len(set(posting_ids))}
    p_keep = np.full(len(texts), np.nan)
    if len(labelled) >= 50 and 0 < y[labelled].mean() < 1:
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=cfg.min_df,
                              max_features=cfg.max_features, sublinear_tf=True)
        # hold out whole postings, so a posting's wording is not on both sides
        rng = np.random.default_rng(cfg.seed)
        postings = sorted({posting_ids[i] for i in labelled})
        held = set(rng.choice(len(postings), size=max(1, int(len(postings) * cfg.test_share)),
                              replace=False).tolist())
        held_ids = {postings[k] for k in held}
        test = np.array([i for i in labelled if posting_ids[i] in held_ids])
        train = np.array([i for i in labelled if posting_ids[i] not in held_ids])
        x_all = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
        clf.fit(x_all[train], y[train])
        if len(test):
            pred = clf.predict(x_all[test])
            tp = int(((pred == 0) & (y[test] == 0)).sum())      # "drop" is the class that matters
            fp = int(((pred == 0) & (y[test] == 1)).sum())
            fn = int(((pred == 1) & (y[test] == 0)).sum())
            report["agreement_with_weak_labels_on_held_out_postings"] = {
                "n_sentences": int(len(test)), "n_postings": len(held_ids),
                "accuracy": round(float((pred == y[test]).mean()), 4),
                "drop_precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "drop_recall": round(tp / (tp + fn), 4) if tp + fn else None,
                "note": "agreement with the headers' own labels, not accuracy against people"}
        clf.fit(x_all[labelled], y[labelled])                    # final model: all weak labels
        p_keep = clf.predict_proba(x_all)[:, 1]
        report["classifier"] = "tf-idf (1-2 words) + logistic regression"
    else:
        report["classifier"] = "none: too few sentences under headers"
    keep, how = [], {"header": 0, "under_keep_header": 0, "under_drop_header": 0,
                     "classifier_keep": 0, "classifier_drop": 0, "kept_without_evidence": 0}
    for i, w in enumerate(weak):
        if w == "header":
            keep.append(False)
            how["header"] += 1
        elif w in KEEP_ZONES:
            keep.append(True)
            how["under_keep_header"] += 1
        elif w in DROP_ZONES:
            keep.append(False)
            how["under_drop_header"] += 1
        elif np.isnan(p_keep[i]):
            keep.append(True)
            how["kept_without_evidence"] += 1
        elif p_keep[i] < cfg.drop_below:
            keep.append(False)
            how["classifier_drop"] += 1
        else:
            keep.append(True)
            how["classifier_keep"] += 1
    report["decisions"] = how
    report["n_kept"] = int(sum(keep))
    report["n_dropped"] = int(len(keep) - sum(keep))
    return {"keep": keep, "weak": weak, "p_keep": [None if np.isnan(p) else float(p) for p in p_keep],
            "report": report}
