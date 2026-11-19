"""SYNTHETIC five-tag sentences with a complete reference — for testing code only.

Until the experts' gold standard exists there is nothing to train or score the
extraction models on. This generator fills templates with words, so every span
of every sentence is known exactly. That is enough to check that a training
loop learns, that scoring is wired correctly, that the router escalates — and
nothing more.

    These sentences are invented. A score on them says the CODE works. It says
    nothing about how well any model reads real job postings, and must never be
    reported as a result.

A share of the fillers (`long_tail`) is deliberately absent from the lexicon,
so the dictionary baseline cannot be perfect here and the router has something
to escalate.

Rows carry `"synthetic": true`. `write_jsonl` refuses to write into the
gold-standard tree.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from gold_standard.iaa import Span

SYNTHETIC_VERSION = "syn-v1"
NOTICE = ("SYNTHETIC: template-generated sentences for testing code. "
          "Not corpus data, not evidence, never to be cited.")
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "DATA" / "fivetag" / "synthetic"

# fillers: (in the lexicon, NOT in the lexicon)
TT = (["Python", "Java", "Docker", "Kubernetes", "React", "PostgreSQL", "MySQL", "Laravel",
       "Spring Boot", "Jenkins", "MongoDB", "TypeScript", "Node.js", "AWS", "Terraform",
       "Flutter", "Redis", "GitLab CI", "Selenium", "Kotlin"],
      ["Zentrix", "QuarkDB", "Lumen.js", "FlowForge", "Nimbus CI", "Orbital SDK", "TideGraph",
       "VantaQL", "Kestrelio", "Pylonix"])
SH_VERB = (["develop", "design", "build", "write", "implement", "test", "deploy", "maintain",
            "configure", "optimize", "integrate", "automate"],
           ["orchestrate", "provision", "benchmark", "containerize", "instrument", "harden"])
SH_OBJECT = ["REST APIs", "web applications", "data pipelines", "microservices",
             "database schemas", "unit tests", "mobile applications", "user interfaces",
             "backend services", "deployment scripts", "search indexes", "payment modules"]
KN = (["data structures", "algorithms", "design patterns", "software architecture",
       "object-oriented programming", "relational database concepts", "Agile principles",
       "testing principles", "microservices architecture", "security fundamentals"],
      ["event sourcing", "domain-driven design", "concurrency control", "the CAP theorem",
       "eventual consistency", "zero-trust networking"])
SS = (["communication", "teamwork", "leadership", "time management", "problem solving",
       "critical thinking", "negotiation", "collaboration"],
      ["active listening", "stakeholder alignment", "cross-team facilitation",
       "expectation setting"])
WB = (["proactive", "detail-oriented", "disciplined", "adaptable", "responsible", "curious",
       "reliable", "self-motivated"],
      ["meticulous", "self-driven", "accountable", "persistent", "level-headed"])

SH_VERB_ID = (["mengembangkan", "merancang", "membangun", "menguji", "memelihara",
               "mengintegrasikan"], ["mengorkestrasi", "menyediakan"])
SH_OBJECT_ID = ["aplikasi web", "layanan backend", "basis data", "antarmuka pengguna",
                "aplikasi mobile", "skrip otomasi"]
KN_ID = (["struktur data", "algoritma", "konsep microservices", "prinsip SOLID",
          "desain sistem"], ["konsistensi eventual", "pemodelan domain"])
SS_ID = (["komunikasi", "kerja sama tim", "kepemimpinan", "manajemen waktu",
          "pemecahan masalah"], ["mendengarkan aktif", "penyelarasan pemangku kepentingan"])
WB_ID = (["teliti", "disiplin", "proaktif", "bertanggung jawab", "mandiri", "jujur"],
         ["gigih", "berintegritas"])

# A template is a list of pieces: plain text, or (slot, tag). Slots ending in
# "_ID" draw Indonesian fillers. "SHX" is a verb + technology + "clusters"
# phrase: one SH span with a TT span nested inside it (guideline 6.9).
T = [
    ("en", [("SH", "SH"), " using ", ("TT", "TT"), " and ", ("TT", "TT"), "."]),
    ("en", ["Experience with ", ("TT", "TT"), ", ", ("TT", "TT"), ", and ", ("TT", "TT"), "."]),
    ("en", ["Strong understanding of ", ("KN", "KN"), "."]),
    ("en", ["We need someone ", ("WB", "WB"), " and ", ("WB", "WB"), "."]),
    ("en", ["Excellent ", ("SS", "SS"), " skills."]),
    ("en", ["You will ", ("SH", "SH"), " for our clients."]),
    ("en", ["Knowledge of ", ("KN", "KN"), " and ", ("KN", "KN"), " is a plus."]),
    ("en", ["Ability to ", ("SH", "SH"), "."]),
    ("en", ["Familiar with ", ("TT", "TT"), " or ", ("TT", "TT"), "."]),
    ("en", [("SS", "SS"), " and ", ("SS", "SS"), " are essential."]),
    ("en", [("SHX", "SH"), "."]),
    ("en", ["The ideal candidate is ", ("WB", "WB"), ", with good ", ("SS", "SS"), "."]),
    ("en", ["You will ", ("SH", "SH"), " in ", ("TT", "TT"), " and understand ", ("KN", "KN"), "."]),
    ("en", ["Our office is located in Jakarta."]),
    ("en", ["We offer a competitive salary and benefits."]),
    ("en", ["Minimum 3 years of experience in a similar role."]),
    ("en", ["Please send your application before the end of the month."]),
    ("id", ["Menguasai ", ("TT", "TT"), " dan ", ("TT", "TT"), "."]),
    ("id", ["Mampu ", ("SH_ID", "SH"), " menggunakan ", ("TT", "TT"), "."]),
    ("id", ["Memahami ", ("KN_ID", "KN"), "."]),
    ("id", ["Memiliki kemampuan ", ("SS_ID", "SS"), " yang baik."]),
    ("id", ["Pribadi yang ", ("WB_ID", "WB"), " dan ", ("WB_ID", "WB"), "."]),
    ("id", ["Berpengalaman dalam ", ("TT", "TT"), ", ", ("TT", "TT"), ", dan ", ("TT", "TT"), "."]),
    ("id", ["Lokasi kerja di Bandung."]),
    ("id", ["Kirim lamaran Anda sebelum akhir bulan."]),
]

_POOLS = {"TT": TT, "KN": KN, "SS": SS, "WB": WB, "KN_ID": KN_ID, "SS_ID": SS_ID, "WB_ID": WB_ID}


def _pick(rng: random.Random, pool: tuple[list, list], long_tail: float, used: set) -> str:
    for _ in range(50):
        src = pool[1] if (pool[1] and rng.random() < long_tail) else pool[0]
        w = rng.choice(src)
        if w not in used:
            used.add(w)
            return w
    return w


def _fill(rng: random.Random, slot: str, long_tail: float, used: set) -> tuple[str, list]:
    """(text, nested spans relative to the text)."""
    if slot == "SH":
        return f"{_pick(rng, SH_VERB, long_tail, used)} {rng.choice(SH_OBJECT)}", []
    if slot == "SH_ID":
        return f"{_pick(rng, SH_VERB_ID, long_tail, used)} {rng.choice(SH_OBJECT_ID)}", []
    if slot == "SHX":
        verb = rng.choice(["configure", "deploy", "maintain"])
        tech = _pick(rng, TT, long_tail, used)
        return f"{verb} {tech} clusters", [(len(verb) + 1, len(verb) + 1 + len(tech), "TT")]
    return _pick(rng, _POOLS[slot], long_tail, used), []


def generate(n: int, seed: int = 0, long_tail: float = 0.3) -> list[dict]:
    """`n` synthetic rows. Same arguments, same rows."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        t_index = rng.randrange(len(T))
        lang, pieces = T[t_index]
        text, spans, used = "", [], set()
        for piece in pieces:
            if isinstance(piece, str):
                text += piece
                continue
            slot, tag = piece
            filler, nested = _fill(rng, slot, long_tail, used)
            if not text:
                filler = filler[0].upper() + filler[1:]
            start = len(text)
            text += filler
            spans.append({"start": start, "end": len(text), "label": tag})
            spans += [{"start": start + a, "end": start + b, "label": lab} for a, b, lab in nested]
        rows.append({"sentence_id": f"syn-{seed}-{i:05d}", "text": text, "lang": lang,
                     "spans": sorted(spans, key=lambda s: (s["start"], s["end"])),
                     "group": f"template-{t_index:02d}", "synthetic": True,
                     "synthetic_version": SYNTHETIC_VERSION})
    return rows


def as_reference(rows: list[dict]) -> tuple[dict[str, str], dict[str, list[Span]]]:
    """(sentence text by id, reference spans by id) — the shape `evaluate` takes."""
    return ({r["sentence_id"]: r["text"] for r in rows},
            {r["sentence_id"]: [Span(s["start"], s["end"], s["label"]) for s in r["spans"]]
             for r in rows})


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path = Path(path).resolve()
    gold = (REPO_ROOT / "DATA" / "gold_standard").resolve()
    if gold in path.parents:
        raise ValueError("synthetic rows must not be written under DATA/gold_standard/")
    if not all(r.get("synthetic") for r in rows):
        raise ValueError("write_jsonl only writes rows marked synthetic")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_notice": NOTICE, "synthetic_version": SYNTHETIC_VERSION},
                            ensure_ascii=False) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a span file (synthetic or exported gold); the notice line is skipped."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if "_notice" not in r:
                    rows.append(r)
    return rows
