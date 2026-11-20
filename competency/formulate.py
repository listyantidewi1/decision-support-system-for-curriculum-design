"""Competency statements formulated the way SKKNI units are titled, from the market.

The researcher (2026-11-20), correcting the developer: the SKKNI units the
promotors showed are EXAMPLES OF HOW A COMPETENCY STATEMENT IS FORMULATED, not a
format to translate the market's competencies into. SKKNI is part of the
problem the study addresses: the programming standard dates from 2016, and the
study's aim is to follow what industry asks for now. The competencies come from
current postings. Where one of them IS an SKKNI unit, the SKKNI wording may be
used.

The promotors: a task title ("develop web application") is a duty; a
competency says how well it is done, which skills or standards apply, and what
outcome it achieves.

So a statement here is one line in Indonesian, shaped like an SKKNI unit title
("Menulis kode dengan prinsip sesuai guidelines dan best practices"), with an
English gloss:

    <operational verb on the competency's level> <the specific object of work>
        [<standard or method: "sesuai prinsip REST", "dengan prinsip OOP">]
        [<quality: "yang aman dan skalabel">]  [<outcome: "untuk ...">]

A bracketed part enters only when the competency's own postings ask for it
often enough (`choose_qualifiers`). No tool or product name enters a statement
(the researcher, 2026-11-20: tools are listed beside a competency).

Steps:
  1. QUALIFIERS (free): `competency.qualifiers.profile`, then the standards,
     qualities and outcome asked for in at least `min_share` of the
     competency's postings.
  2. SKKNI CANDIDATES (free): the `n_candidates` SKKNI units nearest to the
     competency by multilingual sentence embedding, over every SKKNI document
     in DATA/master_skkni (`competency.skkni`), whatever its year.
  3. WRITE (paid, capped): a model formulates the statement from these slots and
     says whether one candidate is the SAME competency (sama), overlaps it
     (sebagian) or none is (tidak).
  4. CHECK (free): refuses a statement whose verb is not on the competency's
     level, that names a tool, a number, or a standard or quality the slots did
     not give; an SKKNI code outside the candidates; an SKKNI-worded statement
     that does not begin with the SKKNI title as written. One retry.
  5. WORDING: the SKKNI title is used as written (optionally followed by one
     standard or outcome of the slots) when the competency is the SAME as an
     SKKNI unit AND the title's head verb is on the competency's level. When the
     levels differ the market's statement is kept, and the SKKNI unit is
     reported with both levels: the market asks the same work at another level
     than the standard states, which is itself a finding.
  6. CONFIRM (paid, capped, only for a 'sama'): a second model of another
     family (`CONFIRM_MODEL`), shown the competency's evidence and that one unit
     and nothing the writer decided, must also say 'sama'. Otherwise the call
     becomes 'sebagian' and the market's statement stays (`apply_confirmation`).
  7. MARK (paid, capped; the researcher's decision of 2026-11-21: mark, do not
     leave out): a focused call (`MARK_MODEL`), shown only the evidence, marks
     a competency that is not work (`bukan_pekerjaan`: a work behaviour, a
     purpose clause, employer text, a cut phrase) or a family of different
     works (`keluarga_campuran`), with a reason. One whose postings share very
     few distinct sentences is marked `kalimat_berulang` (free). Marks go into
     `flags`, beside the statement, and change no status; what is marked not
     work keeps no SKKNI link.

Since version 5 a qualifier that fits the object is written (the researcher's
decision of 2026-11-21: a statement names the standard or quality its postings
ask for); when none fits, the writer must say why (`alasan_tanpa_kualifikasi`).

"Same competency" was checked against three independent judges (EXP-032), not
against experts.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from .qualifiers import QUALITY_TERMS, NAMED_STANDARDS, profile, restates
from .statement import INDONESIAN_VERBS, LEVELS, VERB_INFO, VERBS, is_fragment, read_phrase

FORMULATE_VERSION = "formulate-v5"
# the second, independent decision a 'sama' needs before SKKNI wording is used:
# another model family than the writer (GPT) and than the reference judges (Claude)
CONFIRM_MODEL = "qwen/qwen3.8-flash"
BLOOM_CODE = {level: f"C{i}" for i, level in enumerate(LEVELS, 1)}
RELATIONS = ("sama", "sebagian", "tidak")
NO_LEVEL_VERBS_ID = ("melakukan", "melaksanakan")
NO_LEVEL_VERBS_EN = ("perform", "carry")
# The promotors' examples (SKKNI J.620100, 2016): how a statement is formulated.
STYLE_EXAMPLES = (
    "Menggunakan struktur data",
    "Mengimplementasikan user interface",
    "Melakukan instalasi software tools pemrograman",
    "Menulis kode dengan prinsip sesuai guidelines dan best practices",
    "Mengimplementasikan pemrograman terstruktur",
    "Menggunakan library atau komponen pre-existing",
    "Membuat dokumen kode program",
)
# objects too general to name a competency by themselves
GENERIC_OBJECTS = {
    "application", "applications", "software", "solution", "solutions", "system", "systems",
    "feature", "features", "product", "products", "service", "services", "project", "projects",
    "process", "processes", "tool", "tools", "platform", "platforms", "development", "work",
    "aplikasi", "solusi", "sistem", "fitur", "produk", "layanan", "proyek", "proses", "pekerjaan"}
# Indonesian quality words, for the check that a statement adds no quality the slots lack
_QUALITY_ID = {"aman": "security", "andal": "reliability", "efisien": "performance efficiency",
               "skalabel": "performance efficiency", "responsif": "portability",
               "stabil": "reliability", "cepat": "performance efficiency",
               "mudah dipelihara": "maintainability", "mudah digunakan": "usability",
               "akurat": "functional suitability", "lintas platform": "portability",
               "tangguh": "reliability", "berkinerja tinggi": "performance efficiency"}
# English quality words -> the Indonesian a statement uses; and ordinary words
# that must not stay English in an Indonesian statement (the first sample mixed
# them: "Memelihara data pipelines yang scalable", EXP-032)
GLOSSARY = {
    "scalable": "skalabel", "scalability": "skalabilitas", "reliable": "andal",
    "reliability": "keandalan", "secure": "aman", "security": "keamanan",
    "maintainable": "mudah dipelihara", "maintainability": "kemudahan pemeliharaan",
    "responsive": "responsif", "efficient": "efisien", "robust": "tangguh", "fast": "cepat",
    "high-performance": "berkinerja tinggi", "cross-platform": "lintas platform",
    "user-friendly": "mudah digunakan", "accessible": "mudah diakses", "modular": "modular",
    "reusable": "dapat digunakan ulang", "clean": "bersih", "stable": "stabil",
    "requirements": "kebutuhan", "issues": "masalah", "production": "lingkungan produksi",
    "code": "kode", "database": "basis data", "databases": "basis data",
    "data pipelines": "pipeline data", "automated testing": "pengujian otomatis",
    "performance": "kinerja", "applications": "aplikasi", "web applications": "aplikasi web",
    "mobile applications": "aplikasi mobile", "features": "fitur", "systems": "sistem",
    "solutions": "solusi", "processes": "proses", "data quality": "kualitas data",
    "business requirements": "kebutuhan bisnis", "documentation": "dokumentasi",
    "test plans": "rencana pengujian", "test plan": "rencana pengujian",
    "production issues": "masalah di lingkungan produksi", "monitoring tools": "alat pemantauan",
    "design documents": "dokumen desain", "accessible": "aksesibel (memenuhi aksesibilitas)",
    "accessibility": "aksesibilitas", "programming": "pemrograman", "development": "pengembangan",
    "maintenance": "pemeliharaan",
    "deliver": "menerapkan (C3) atau menghasilkan (C6), setingkat kompetensinya; bukan 'penyampaian'",
    "debugging issues": "debugging", "write code": "menulis kode",
    "prototypes": "purwarupa", "releases": "rilis", "documents": "dokumen",
    "trunk-based development": "trunk-based development", "query": "kueri", "queries": "kueri"}
_MUST_TRANSLATE = {"scalable", "reliable", "secure", "maintainable", "responsive", "efficient",
                   "robust", "accessible", "requirements", "issues", "production", "code",
                   "databases", "pipelines", "performance", "applications", "features",
                   "systems", "solutions", "processes", "scalability", "reliability",
                   "automation", "documentation", "tools", "testing", "support",
                   "programming", "development", "maintenance", "plans", "documents",
                   "designs", "prototypes", "releases", "query", "queries"}
# Indonesian verbs the glossary gives for a posting verb that has no Indonesian verb of its
# own ("deliver"): the checker counts them as that verb
_RENDERS = {"menerapkan": ("deliver",), "menghasilkan": ("deliver",)}
_ENGLISH_ALLOWED_AFTER = {"code review", "source code", "unit test", "best practices",
                          "user interface", "trunk-based development", "trunk based development",
                          "test-driven development", "test driven development"}


_STANDARD_RX = [(name, re.compile(rx, re.I if not name.startswith(("SOLID", "OWASP", "MVC", "REST"))
                                  else 0)) for name, rx in NAMED_STANDARDS]
# the Indonesian forms a statement may give a named standard; used both to see that
# a standard listed as used is written, and to refuse one the slots did not give
_STANDARD_ID_RX = {name: re.compile(rx, re.I) for name, rx in {
    "best practices": r"praktik terbaik",
    "coding standards": r"standar (?:pengodean|penulisan kode|kode|koding)|konvensi (?:penulisan )?kode",
    "design patterns": r"pola (?:desain|perancangan)",
    "clean code": r"kode (?:yang )?bersih",
    "object-oriented principles": r"berorientasi objek|\bPBO\b",
    "test-driven development (TDD)": r"berbasis pengujian",
    "accessibility standards (WCAG)": r"standar aksesibilitas",
    "code review": r"(?:peninjauan|tinjauan|penelaahan) kode",
    "unit tests": r"pengujian unit",
    "security standards": r"standar keamanan",
    "industry standards": r"standar industri",
    "SDLC": r"siklus hidup pengembangan",
    "documentation standards": r"standar dokumentasi",
    "company standards": r"standar (?:perusahaan|organisasi|internal|tim)",
}.items()}


def _names_standard(name: str, text: str) -> bool:
    rx = dict(_STANDARD_RX).get(name)
    id_rx = _STANDARD_ID_RX.get(name)
    return bool((rx is not None and rx.search(text)) or (id_rx is not None and id_rx.search(text)))


def _says(q: dict, text: str) -> bool:
    """Does a statement actually write qualifier q, in English or in Indonesian?"""
    low = text.lower()

    def has(word: str) -> bool:
        return bool(word) and bool(re.search(rf"(?<![\w-]){re.escape(word.lower())}(?![\w-])", low))
    if q["kind"] == "quality":
        char = q.get("iso25010")
        words = {q["text"]} | {w for w, c in QUALITY_TERMS.items() if c == char} \
            | {w for w, c in _QUALITY_ID.items() if c == char}
        if GLOSSARY.get(q["text"].lower()):
            words.add(GLOSSARY[q["text"].lower()].split(" (")[0])
        return any(has(w) for w in words)
    if q["kind"] == "standard":
        name = q["text"].replace("following: ", "")
        return _names_standard(q["text"], text) or has(name)
    if q["kind"] == "outcome":
        return has("untuk") or has("guna") or has("agar")
    return True


@dataclass(frozen=True)
class FormulateConfig:
    min_share: float = 0.05           # a standard or quality enters when this share of postings asks
    min_postings: int = 3
    max_standards: int = 2
    max_qualities: int = 2
    outcome_min_postings: int = 5
    outcome_min_share: float = 0.02
    n_candidates: int = 8             # the same 8 nearest units the reference judges saw (EXP-032)
    max_tokens: int = 700
    # the second model reasons before it answers; 1,200 tokens cut 4 of 14 answers off (EXP-032)
    confirm_max_tokens: int = 4000
    # marked "kalimat_berulang" when the postings share this few distinct sentences
    # (distinct / postings): one employer's text, not a skill. On e0_pool_all_v8 the 10
    # review candidates at or below 0.3 were all employer text ("Apply today",
    # "Support their families", "Create inclusive space") (EXP-034)
    repeated_sentence_ratio: float = 0.3
    repeated_sentence_min_postings: int = 5
    version: str = FORMULATE_VERSION


# ------------------------------------------------------------------ the slots
def choose_qualifiers(prof: dict, cfg: FormulateConfig | None = None) -> list[dict]:
    """The standards, qualities and outcome a statement may carry, each with its
    profile id and postings. Named standards come before a generic "following
    ..." phrase; a quality without a characteristic ("high-quality") is too
    vague to say how well; one word per quality characteristic."""
    cfg = cfg or FormulateConfig()
    out: list[dict] = []
    std = [s for s in prof["standards"]
           if s["share"] >= cfg.min_share and s["n_postings"] >= cfg.min_postings]
    std.sort(key=lambda s: (s["text"].startswith("following: "), -s["n_postings"], s["text"]))
    for s in std[:cfg.max_standards]:
        out.append({**s, "kind": "standard", "text": s["text"].replace("following: ", "")})
    seen: set[str] = set()
    for q in sorted(prof["qualities"], key=lambda q: (-q["n_postings"], q["text"])):
        char = q.get("iso25010", "general")
        if (char == "general" or char in seen or q["share"] < cfg.min_share
                or q["n_postings"] < cfg.min_postings):
            continue
        seen.add(char)
        out.append({**q, "kind": "quality"})
        if len(seen) >= cfg.max_qualities:
            break
    for r in prof["outcomes"]:
        if r["n_postings"] >= cfg.outcome_min_postings and r["share"] >= cfg.outcome_min_share:
            out.append({**r, "kind": "outcome"})
            break
    return out


def allowed_verbs(level: str | None) -> tuple[list[str], list[str]]:
    """Indonesian and English verbs of the verb table ON the level (for a
    competency without a level: the verbs that only say "do it")."""
    if level not in LEVELS:
        return list(NO_LEVEL_VERBS_ID), ["perform", "carry out"]
    idn = sorted(v for v, r in VERB_INFO.items() if r["lang"] == "id" and r["level"] == level
                 and r["basis"] in ("ID", "NOM") and v not in ("melakukan", "melaksanakan"))
    en = sorted(v for v, r in VERB_INFO.items() if r["lang"] == "en" and r["level"] == level
                and r["basis"] in ("AK", "LIST", "DEV"))
    return idn, en


# ------------------------------------------------------------- SKKNI candidates
def unit_text(u: dict) -> str:
    parts = [u["title"], u.get("title_en") or "", (u.get("description") or "")[:400],
             "; ".join(e["title"] for e in u.get("elements", []))]
    return ". ".join(p for p in parts if p)


class SkkniIndex:
    """Every SKKNI unit, embedded once, to find the nearest units of a competency."""

    def __init__(self, units: Sequence[dict], embed: Callable | None = None):
        import numpy as np
        self.units = list(units)
        self._embed = embed or _default_embed
        vec = np.asarray(self._embed([unit_text(u) for u in self.units]), dtype=np.float32)
        self.vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)

    def nearest(self, text: str, k: int) -> list[tuple[dict, float]]:
        import numpy as np
        q = np.asarray(self._embed([text]), dtype=np.float32)[0]
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self.vec @ q
        order = np.argsort(-sims)[:k]
        return [(self.units[int(i)], round(float(sims[int(i)]), 4)) for i in order]


def _default_embed(texts):
    from .mapping import shared_embedder
    return shared_embedder().encode(list(texts), batch_size=64, show_progress_bar=False,
                                    normalize_embeddings=True)


def competency_text(prof: dict) -> str:
    objects = "; ".join(o["text"] for o in prof["objects"][:4])
    phrases = "; ".join(prof["example_phrases"][:6])
    return f"{prof['statement']}. {objects}. {phrases}"


def candidate_rows(cands: Sequence[tuple[dict, float]]) -> list[dict]:
    return [{"kode": u["code"], "judul": u["title"], "judul_en": u.get("title_en", ""),
             "tahun": u.get("decree_year", ""), "deskripsi": (u.get("description") or "")[:300],
             "elemen": [e["title"] for e in u.get("elements", [])][:6], "kemiripan": sim}
            for u, sim in cands]


# --------------------------------------------------------------------- writing
# One definition of 'sama' for the writer (rule 4) and for the confirmation, so
# the two cannot drift apart. It is the reference judges' definition (EXP-032).
SAME_RULE = (
    "'sama' HANYA jika satu unit adalah kompetensi "
    "yang SAMA: pekerjaan yang sama pada objek yang sama dengan cakupan yang sama, sehingga ahli "
    "akan memetakannya satu-satu (kata, rincian, dan tingkat boleh berbeda). 'sebagian' jika "
    "tumpang tindih tetapi objek atau cakupannya berbeda, atau yang satu hanya bagian dari yang "
    "lain (memelihara basis data vs. menyelesaikan masalah kinerja basis data; mengembangkan "
    "aplikasi web vs. aplikasi IoT). 'tidak' jika tidak ada yang tumpang tindih berarti. Unit dari "
    "bidang lain (game, multimedia, IoT, pertolongan pertama) jarang 'sama' untuk pekerjaan "
    "perangkat lunak umum. Bersikaplah ketat terhadap 'sama'.")

SYSTEM = (
    "Anda merumuskan SATU pernyataan kompetensi untuk kurikulum SMK Rekayasa Perangkat Lunak "
    "dari bukti lowongan kerja, dengan GAYA PERUMUSAN judul unit SKKNI (contoh diberikan). "
    "Kompetensinya berasal dari lowongan, bukan dari SKKNI. Aturan:\n"
    "1. pernyataan (Bahasa Indonesia): satu baris pendek seperti judul unit SKKNI: kata kerja "
    "operasional + objek kerja yang spesifik + standar/metode, mutu, atau hasil dari 'kualifikasi'. "
    "Jika 'kualifikasi' berisi yang cocok dengan objek pernyataan, MASUKKAN yang paling sering "
    "(paling banyak dua): itulah 'seberapa baik' dan 'standar apa' yang diminta lowongan. Jika tidak "
    "ada yang cocok, tulis alasannya di 'alasan_tanpa_kualifikasi'. Jangan menambah standar, mutu, "
    "hasil, angka, atau ambang yang tidak ada di 'kualifikasi'.\n"
    "1a. KATA KERJA: pakai kata kerja yang DIPAKAI LOWONGAN ('kata_kerja_utama', yang paling "
    "sering), dalam bentuk Indonesianya, atau 'Melakukan'/'Melaksanakan' + kata benda kegiatannya "
    "('Melakukan debugging', 'Melaksanakan pengujian unit'). JANGAN menggantinya dengan kata kerja "
    "lain yang kebetulan setingkat (debug tidak boleh menjadi 'menganalisis'; monitor tidak boleh "
    "menjadi 'mengevaluasi'). Jika lowongan menyebut kegiatan itu sebagai kata benda "
    "('programming', 'pemrograman', 'testing'), tulis 'Melakukan pemrograman' / 'Perform "
    "programming'. Dokumen dibuat atau disusun, bukan 'dilakukan'.\n"
    "1b. BAHASA: kata umum dalam bahasa Indonesia baku (lihat 'glosarium'); hanya istilah teknis "
    "yang lazim dalam SKKNI boleh tetap Inggris: API, user interface, code review, debugging, unit "
    "test, library, framework, source code, best practices, guidelines, REST, CI/CD, DevOps, "
    "deployment.\n"
    "1c. MUTU menerangkan ARTEFAK (aplikasi, sistem, kode, API, pipeline, basis data), bukan "
    "kebutuhan, kinerja, masalah, atau standar: 'Mengembangkan aplikasi web yang aman', bukan "
    "'kebutuhan bisnis yang skalabel'. Jika mutu tidak dapat diletakkan pada artefak pernyataan "
    "itu, jangan dipakai. Standar ditulis 'sesuai prinsip REST', 'sesuai best practices'; hasil "
    "ditulis 'untuk meningkatkan kinerja'.\n"
    "1d. BACA FRASA DENGAN BENAR: 'monitoring tools' adalah alat pemantauan (kata benda), bukan "
    "'memantau alat': yang dipantau adalah sistem; 'test plans' adalah rencana pengujian yang "
    "disusun; 'deliver solutions' adalah menerapkan solusi (C3) atau menghasilkan solusi (C6), "
    "setingkat kompetensinya, bukan 'penyampaian'.\n"
    "1e. TANPA TINGKAT ('tidak dinyatakan'): 'Melakukan'/'Melaksanakan' + kata benda KEGIATAN "
    "(penyusunan, pembuatan, pengujian, pemantauan, pendokumentasian): 'Melakukan penyusunan "
    "rencana pengujian', bukan 'Melakukan test plans' atau 'Melakukan dokumen desain'.\n"
    "1f. JANGAN menambah objek atau konteks yang tidak ada di bukti ('aplikasi', 'perangkat "
    "lunak', 'pada pemrograman', 'masalah', 'teknis'); jika objek di bukti umum, biarkan umum.\n"
    "2. JANGAN menyebut nama alat, produk, bahasa pemrograman, atau framework (Python, React, AWS, "
    "Laravel, dll.); alat dicatat terpisah.\n"
    "3. statement_en: terjemahan Inggris yang setia dari 'pernyataan', dalam bentuk perintah "
    "('Develop ...', 'Perform ...'). kualifikasi_dipakai: id "
    "kualifikasi yang benar-benar tertulis dalam pernyataan; kosong jika tidak ada.\n"
    "4. skkni: bandingkan dengan 'kandidat_skkni'. " + SAME_RULE + "\n"
    "5. Jika 'sama': pernyataan_skkni = judul unit SKKNI itu PERSIS seperti tertulis; jika "
    "'kualifikasi' berisi standar atau hasil yang cocok, tambahkan SATU (yang paling sering) yang "
    "tersambung wajar ('Melakukan code "
    "review sesuai best practices', 'Melaksanakan pengujian unit program dalam alur CI/CD'). "
    "JANGAN menambahkan mutu ('yang ...') pada judul SKKNI. pernyataan_skkni_en = terjemahan "
    "Inggrisnya.\n"
    "Jawab HANYA dengan JSON: {\"pernyataan\":\"\",\"statement_en\":\"\","
    "\"kualifikasi_dipakai\":[],\"alasan_tanpa_kualifikasi\":\"\","
    "\"skkni\":{\"kode\":null,\"hubungan\":\"tidak\",\"alasan\":\"\"},"
    "\"pernyataan_skkni\":null,\"pernyataan_skkni_en\":null}")


def main_verbs(prof: dict, n: int = 3) -> list[str]:
    """The verbs the competency's postings use most, on its level."""
    level = prof.get("level")
    return [v["text"] for v in prof["verbs"] if VERBS.get(v["text"]) == level][:n]


def build_messages(prof: dict, chosen: Sequence[dict], cands: Sequence[tuple[dict, float]],
                   problems: Sequence[str] = ()) -> list[dict]:
    idn, en = allowed_verbs(prof.get("level"))
    level = (f"{prof['code']} {prof['level']}" if prof.get("level")
             else "tidak dinyatakan (lowongan tidak memakai kata kerja)")
    payload = {
        "contoh_gaya_skkni": list(STYLE_EXAMPLES),
        "tingkat_kognitif": level,
        "kata_kerja_utama": [{"verb": v, "n": next(x["n_postings"] for x in prof["verbs"]
                                                     if x["text"] == v)} for v in main_verbs(prof)],
        "kata_kerja_id": idn, "kata_kerja_en": en,
        "kompetensi": {
            "pernyataan_aturan_en": prof["statement"], "n_lowongan": prof["n_postings"],
            "kata_kerja_lowongan": [{"verb": v["text"], "n": v["n_postings"]} for v in prof["verbs"][:6]],
            "objek_kerja": [{"teks": o["text"], "n": o["n_postings"]} for o in prof["objects"][:4]],
            "contoh_frasa": prof["example_phrases"][:8]},
        "kualifikasi": [{"id": q["id"], "jenis": q["kind"], "teks": q["text"],
                         "n_lowongan": q["n_postings"]} for q in chosen],
        "kandidat_skkni": candidate_rows(cands),
        "glosarium": GLOSSARY,
    }
    user = json.dumps(payload, ensure_ascii=False)
    if problems:
        user += ("\n\nJAWABAN SEBELUMNYA DITOLAK karena:\n- " + "\n- ".join(problems)
                 + "\nPerbaiki dan jawab ulang dengan JSON lengkap.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------------- checking
# A statement's qualifier clause ("sesuai prinsip REST", "according to REST principles",
# "yang aman") is not part of its work: the verb and its level are read from what comes
# before it. Without the cut, "Perform API testing according to REST principles" read as
# 'perform' (C3) instead of testing (C5) (EXP-032).
_QUALIFIER_START = re.compile(
    r"\s+(?:sesuai|dalam|untuk|dengan|yang|berdasarkan|menggunakan|secara|according\s+to|"
    r"following|in\s+line\s+with|in\s+accordance\s+with|with|to|that|for|using)\s+", re.I)


def work_part(text: str) -> str:
    m = _QUALIFIER_START.search(text)
    return text[:m.start()] if m else text


def levels_of(text: str) -> list[str]:
    r = read_phrase(work_part(text))
    return [VERBS[v] for v in r.verbs if v in VERBS and r.how != "none"]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w/+#.\- ]", " ", text.lower()).split())


def sentence_case(text: str) -> str:
    """One casing for every statement, as in the promotors' examples ("Menggunakan
    struktur data"): first letter capital, other words lower case, except
    acronyms and mixed-case terms (API, SQL, CI/CD, IoT, DevOps)."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        core = w.strip("()[],.;:")
        keep = (any(ch.isdigit() for ch in core) or "/" in core
                or (len(core) >= 2 and core.isupper())
                or any(ch.isupper() for ch in core[1:]))
        out.append(w if keep else w.lower())
    return _capital(" ".join(out))


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _first_word(text: str) -> str:
    w = text.strip().split()
    return w[0].lower().strip(",.") if w else ""


def check(raw: str | None, prof: dict, chosen: Sequence[dict],
          cands: Sequence[tuple[dict, float]]) -> tuple[dict | None, list[str]]:
    """(answer, problems). An answer with any problem is not accepted."""
    try:
        a = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip()))
    except (TypeError, ValueError):
        return None, ["the answer is not JSON"]
    if not isinstance(a, dict):
        return None, ["the answer is not a JSON object"]
    problems: list[str] = []
    level = prof.get("level")
    pid, pen = str(a.get("pernyataan") or "").strip(), str(a.get("statement_en") or "").strip()
    if not pid or not pen:
        problems.append("pernyataan and statement_en are both required")
    a["pernyataan"], a["statement_en"] = _capital(pid), _capital(pen)
    if a.get("pernyataan_skkni"):
        a["pernyataan_skkni"] = _capital(str(a["pernyataan_skkni"]).strip())
    low = " " + pid.lower() + " "
    for kept in _ENGLISH_ALLOWED_AFTER:
        low = low.replace(" " + kept + " ", " ")
    mixed = sorted(w for w in _MUST_TRANSLATE if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low))
    if mixed:
        problems.append(f"pernyataan mixes English words that have an Indonesian form: {mixed} "
                        f"(see the glossary)")
    if level:
        if pid and level not in levels_of(pid):
            problems.append(f"pernyataan '{pid}' does not open with a verb on the level '{level}' "
                            f"(read: {levels_of(pid) or 'no verb'})")
        if pen and level not in levels_of(pen):
            problems.append(f"statement_en '{pen}' does not open with a verb on the level '{level}'")
    else:
        if pid and _first_word(pid) not in NO_LEVEL_VERBS_ID:
            problems.append(f"the postings state no level: pernyataan must open with "
                            f"{' or '.join(NO_LEVEL_VERBS_ID)}")
        if pen and _first_word(pen) not in NO_LEVEL_VERBS_EN:
            problems.append("the postings state no level: statement_en must open with 'Perform'")
    used = [str(x) for x in a.get("kualifikasi_dipakai") or []]
    chosen_ids = {q["id"] for q in chosen}
    if set(used) - chosen_ids:
        problems.append(f"kualifikasi_dipakai names ids that were not given: {sorted(set(used) - chosen_ids)}")
    # how well / which standard: a fitting qualifier is written, or the writer says why
    # none fits (the writer left one out in 8 of 19 held-out cases, EXP-032; the
    # researcher's decision of 2026-11-21)
    if chosen and not used and not str(a.get("alasan_tanpa_kualifikasi") or "").strip():
        problems.append("'kualifikasi' offers a standard, quality or outcome but none is used: "
                        "write the most frequent one that fits the object, or give "
                        "'alasan_tanpa_kualifikasi'")
    # a quality is not a standard: "sesuai prinsip efisien" is refused, "kueri yang efisien"
    # is meant (a v5 dry run)
    qual_words = sorted(set(_QUALITY_ID) | {w for w, c in QUALITY_TERMS.items() if c != "general"},
                        key=len, reverse=True)
    as_standard = re.compile(r"\b(?:sesuai|according\s+to|following)\s+(?:(?:prinsip|principles?\s+of"
                             r"|the)\s+)?(?:" + "|".join(map(re.escape, qual_words)) + r")\b", re.I)
    # (the Indonesian line only: "following secure coding practices" is a fair English gloss)
    if as_standard.search(pid):
        problems.append("pernyataan writes a quality as if it were a standard ('sesuai ... "
                        "<quality>'): put the quality on the artefact ('... yang <mutu>')")
    # a qualifier listed as used must be written: the line, or the SKKNI wording, says it
    # (a dry run of v5 listed 'efficient' for "Memelihara query basis data")
    said = pid + " " + str(a.get("pernyataan_skkni") or "")
    for q in chosen:
        if q["id"] in set(used) and not _says(q, said):
            problems.append(f"kualifikasi_dipakai lists {q['id']} ('{q['text']}') but the "
                            f"pernyataan does not say it")
    # Slips in the SKKNI fields do not cost the statement: the link is set to the
    # cautious reading and a note is kept. The writer named the nearest unit
    # beside 'tidak', or claimed an overlap without a unit, in 8 of 53 answers (EXP-032).
    sk = dict(a.get("skkni") or {}) if isinstance(a.get("skkni"), dict) else {}
    code, rel = sk.get("kode"), sk.get("hubungan")
    by_code = {u["code"]: u for u, _ in cands}
    notes, slips = [], []
    if rel not in RELATIONS:
        notes.append(f"hubungan {rel!r} tidak dikenal: dibaca 'tidak'")
        slips.append(f"skkni.hubungan must be one of {RELATIONS}")
        rel, code = "tidak", None
    elif rel == "tidak" and code:
        notes.append(f"kode {code} dibuang: hubungan 'tidak'")
        slips.append("skkni.kode must be null when hubungan is 'tidak'")
        code = None
    elif rel in ("sama", "sebagian") and code not in by_code:
        notes.append(f"hubungan '{rel}' tanpa kode kandidat ({code!r}): dibaca 'tidak'")
        slips.append(f"skkni.kode {code!r} is not one of the candidates: give the code of the "
                     f"candidate meant, or hubungan 'tidak'")
        rel, code = "tidak", None
    sk.update(kode=code, hubungan=rel)
    if notes:
        sk["catatan"] = notes
    a["skkni"] = sk
    if slips:
        a["_skkni_slips"] = slips          # formulate_one asks once more before it settles
    sk_text = str(a.get("pernyataan_skkni") or "").strip()
    title = by_code[code]["title"] if code in by_code else ""
    if rel == "sama" and code in by_code:
        if not sk_text:
            problems.append("hubungan 'sama' needs pernyataan_skkni (the SKKNI title as written)")
        elif not _norm(sk_text).startswith(_norm(title)):
            problems.append(f"pernyataan_skkni must begin with the SKKNI title as written: '{title}'")
        else:
            tail = _norm(sk_text)[len(_norm(title)):].strip()
            if tail and not re.match(r"^(sesuai|dalam|dengan|untuk)\b", tail):
                problems.append("after the SKKNI title only a standard or outcome may follow "
                                "('sesuai ...', 'dalam ...', 'dengan ...', 'untuk ...')")
            if tail and (any(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", tail)
                             for w in list(_QUALITY_ID) + [w for w, c in QUALITY_TERMS.items()
                                                           if c != "general"])
                         or re.search(r"\byang\b", tail)):
                problems.append("no quality may be appended to an SKKNI title ('yang ...')")
        if not str(a.get("pernyataan_skkni_en") or "").strip():
            problems.append("hubungan 'sama' needs pernyataan_skkni_en (the SKKNI title in English)")
    main = main_verbs(prof)
    if level and main and pid and pen:
        used = set(read_phrase(work_part(pid)).verbs) | set(read_phrase(work_part(pen)).verbs)
        used |= {en for v in list(used) for en in _RENDERS.get(v, ())}
        if not used & set(main):
            problems.append(f"the statement does not use a verb the postings use for this work "
                            f"on its level (most used: {main}); do not swap in another verb")
    texts = {"pernyataan": pid, "statement_en": pen, "pernyataan_skkni": sk_text}
    # numbers allowed are those written in the evidence TEXT (a version, "24/7"),
    # never a count of postings
    evidence = " ".join(
        [prof["statement"]] + list(prof["example_phrases"]) + list(prof["example_sentences"])
        + [x["text"] for key in ("verbs", "objects", "qualities", "standards", "outcomes", "tools")
           for x in prof.get(key, [])] + [u["title"] for u, _ in cands])
    numbers_ok = set(re.findall(r"\d+(?:[.,]\d+)?", evidence))
    chosen_chars = {q.get("iso25010") for q in chosen if q["kind"] == "quality"}
    chosen_std = {q["text"] for q in chosen if q["kind"] == "standard"}
    for name, text in texts.items():
        if not text:
            continue
        allowed_tool_text = _norm(title) if name == "pernyataan_skkni" else ""
        for t in prof["tools"]:
            tt = t["text"]
            if len(tt) < 2 or (allowed_tool_text and _norm(tt) in allowed_tool_text):
                continue
            flags = re.I if len(tt) >= 4 else 0
            if re.search(rf"(?<![\w.]){re.escape(tt)}(?![\w])", text, flags):
                problems.append(f"{name} names the tool '{tt}'")
        for num in re.findall(r"\d+(?:[.,]\d+)?", text):
            if num not in numbers_ok:
                problems.append(f"{name} states the number {num}, which the evidence does not hold")
        body = text[len(title):] if name == "pernyataan_skkni" and title else text
        for word, char in QUALITY_TERMS.items():
            if char != "general" and char not in chosen_chars and re.search(
                    rf"(?<![\w-]){re.escape(word)}(?![\w-])", body, re.I):
                problems.append(f"{name} adds the quality '{word}', which the slots did not give")
        for word, char in _QUALITY_ID.items():
            if char not in chosen_chars and re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])",
                                                      body, re.I):
                problems.append(f"{name} adds the quality '{word}', which the slots did not give")
        for sname, rx in _STANDARD_RX:
            # naming the work itself ("code review" for "Review code") adds no standard
            if sname not in chosen_std and _names_standard(sname, body) \
                    and not rx.search(prof["statement"]) and not restates(sname, prof["statement"]):
                problems.append(f"{name} adds the standard '{sname}', which the slots did not give")
    if problems:
        problems += slips                  # a retry should mend the SKKNI fields too
    return (a if not problems else None), sorted(set(problems), key=problems.index)


# --------------------------------------------------------------------- wording
def decide_wording(a: dict, prof: dict, cands: Sequence[tuple[dict, float]]) -> dict:
    by_code = {u["code"]: (u, s) for u, s in cands}
    sk = a.get("skkni") or {}
    code, rel = sk.get("kode"), sk.get("hubungan")
    level = prof.get("level")
    market = sentence_case(a["pernyataan"])
    out = {"pernyataan_pasar": market, "statement_en": a["statement_en"],
           "skkni": None, "pernyataan": market, "sumber_rumusan": "pasar"}
    if code in by_code and rel in ("sama", "sebagian"):
        u, sim = by_code[code]
        title_levels = levels_of(u["title"])
        # the HEAD verb of the title carries its level ("Mengidentifikasi dan Menyelesaikan
        # ..." is on C1, whatever the second verb): EXP-032 found coordinated titles let through
        title_level = title_levels[0] if title_levels else None
        agrees = (level is None and title_level is None) or (level is not None and title_level == level)
        out["skkni"] = {"kode": code, "judul": u["title"], "judul_en": u.get("title_en", ""),
                        "tahun": u.get("decree_year", ""), "dokumen": u.get("source", ""),
                        "hubungan": rel, "alasan": sk.get("alasan", ""), "kemiripan": sim,
                        "tingkat_judul": title_level,
                        "kode_tingkat_judul": BLOOM_CODE.get(title_level) if title_level else None,
                        "tingkat_sama": bool(agrees)}
        if rel == "sama" and agrees:
            # the SKKNI wording as written; the market's qualifiers are shown beside it
            sk_text = str(a.get("pernyataan_skkni") or "").strip()
            tail = sk_text[len(u["title"].strip()):].strip() if _norm(sk_text).startswith(
                _norm(u["title"])) else ""
            out["pernyataan"] = sentence_case(u["title"].strip() + (" " + tail if tail else ""))
            out["statement_en"] = sentence_case(str(a.get("pernyataan_skkni_en") or "").strip()
                                                or u.get("title_en") or a["statement_en"])
            out["statement_en_pasar"] = a["statement_en"]
            out["sumber_rumusan"] = f"SKKNI {code}" + (" + pasar" if tail else "")
    return out


# ---------------------------------------------------------------- confirmation
CONFIRM_SYSTEM = (
    "Anda menilai apakah sebuah kompetensi yang dibaca dari lowongan kerja SAMA dengan SATU unit "
    "SKKNI. " + SAME_RULE + " Nilai cakupan unit dari deskripsi dan elemennya, bukan dari "
    "kemiripan kata judulnya saja.\n"
    "Jawab HANYA dengan JSON: {\"hubungan\":\"sama|sebagian|tidak\",\"alasan\":\"satu kalimat\"}")


def confirm_messages(prof: dict, unit: dict) -> list[dict]:
    """The competency's evidence and ONE unit; nothing of what the writer decided or wrote."""
    payload = {
        "kompetensi": {
            "tingkat_kognitif": (f"{prof['code']} {prof['level']}" if prof.get("level")
                                 else "tidak dinyatakan"),
            "pernyataan_aturan_en": prof["statement"], "n_lowongan": prof["n_postings"],
            "kata_kerja_lowongan": [{"verb": v["text"], "n": v["n_postings"]} for v in prof["verbs"][:6]],
            "objek_kerja": [{"teks": o["text"], "n": o["n_postings"]} for o in prof["objects"][:4]],
            "contoh_frasa": prof["example_phrases"][:8]},
        "unit_skkni": {k: v for k, v in candidate_rows([(unit, 0.0)])[0].items() if k != "kemiripan"},
    }
    return [{"role": "system", "content": CONFIRM_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def confirm_same(prof: dict, unit: dict, confirm: Callable[[list[dict]], str | None],
                 attempts: int = 2) -> dict:
    """The second decision on a 'sama'; asked again once when the answer cannot be
    read (cut off by the token allowance, say). An unreadable answer is not a
    confirmation."""
    problem = ""
    for n in range(1, attempts + 1):
        raw = confirm(confirm_messages(prof, unit))
        try:
            a = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip()))
        except (TypeError, ValueError):
            problem = "the answer is not JSON"
            continue
        rel = a.get("hubungan") if isinstance(a, dict) else None
        if rel not in RELATIONS:
            problem = f"hubungan {rel!r} is not one of {RELATIONS}"
            continue
        return {"hubungan": rel, "alasan": str(a.get("alasan") or "")[:400], "percobaan": n}
    return {"hubungan": None, "alasan": "", "masalah": problem, "percobaan": attempts}


def apply_confirmation(w: dict, conf: dict | None) -> dict:
    """SKKNI wording only when two independent decisions say 'sama'.

    The writer alone was wrong in 4 of its 10 'sama' calls on the development
    sample (EXP-032), and a wrong 'sama' replaces the market's own wording with
    a standard's title. A missed 'sama' costs nothing: the market's wording
    stays. So a 'sama' that is not confirmed (the second decision differs, is
    unreadable, or was not asked) becomes 'sebagian' and keeps the market's
    statement; both decisions are kept for audit."""
    sk = w.get("skkni")
    if not sk or sk.get("hubungan") != "sama":
        return w
    w = dict(w)
    sk = {**sk, "konfirmasi": conf, "dikonfirmasi": bool(conf and conf.get("hubungan") == "sama")}
    w["skkni"] = sk
    if sk["dikonfirmasi"]:
        return w
    sk["hubungan_penulis"] = "sama"
    sk["hubungan"] = "sebagian"
    if w["sumber_rumusan"].startswith("SKKNI"):
        w["pernyataan"] = w["pernyataan_pasar"]
        w["statement_en"] = w.pop("statement_en_pasar", w["statement_en"])
        w["sumber_rumusan"] = "pasar"
    return w


# ----------------------------------------------------------------------- marks
# The researcher's decision of 2026-11-21: competencies that are not work, or that
# mix different works, are MARKED, not left out. A focused call of its own (in a
# v5 dry run the writer, asked to mark as a sixth task, missed "support
# decision-making" although its prompt named that very example).
MARK_MODEL = "qwen/qwen3.8-flash"
MARK_SYSTEM = (
    "Anda memeriksa satu kelompok frasa yang dibaca dari lowongan kerja bidang perangkat lunak, "
    "sebelum kelompok itu dijadikan kompetensi kurikulum SMK Rekayasa Perangkat Lunak. Nilai dari "
    "bukti saja, menurut sebagian besar frasa dan lowongannya.\n"
    "bukan_pekerjaan = true jika bukti TIDAK menyebut pekerjaan perangkat lunak yang dapat "
    "diajarkan dan diamati, yaitu jika frasanya terutama: perilaku atau sikap kerja (bekerja "
    "mandiri, cepat belajar, proaktif); tujuan atau manfaat dari pekerjaan lain tanpa menyebut "
    "pekerjaannya (mendukung pengambilan keputusan, meningkatkan kepuasan pelanggan); uraian, "
    "slogan, atau budaya perusahaan (membangun produk kelas dunia, menciptakan ruang yang "
    "inklusif); tunjangan atau ajakan melamar; pengembangan diri atau karier; tugas di luar "
    "perangkat lunak (rekrutmen, penjualan, administrasi kantor); atau frasa terpotong yang "
    "kehilangan objeknya (menulis unit, mendukung bagian).\n"
    "keluarga_campuran = true jika frasanya menyebut pekerjaan yang BERBEDA-BEDA, bukan satu "
    "pekerjaan dengan variasi kata, sehingga objek kerja yang paling sering hanya mencakup "
    "sebagian kecil lowongan (lihat 'porsi_objek_utama' dan 'n' pada 'objek_kerja').\n"
    "Jawab HANYA dengan JSON: {\"bukan_pekerjaan\":false,\"keluarga_campuran\":false,"
    "\"alasan\":\"satu kalimat\"}")


def mark_messages(prof: dict) -> list[dict]:
    """The competency's evidence only: nothing the writer wrote or decided."""
    payload = {
        "pernyataan_aturan_en": prof["statement"], "n_lowongan": prof["n_postings"],
        "kata_kerja_lowongan": [{"verb": v["text"], "n": v["n_postings"]} for v in prof["verbs"][:6]],
        "objek_kerja": [{"teks": o["text"], "n": o["n_postings"]} for o in prof["objects"][:6]],
        "porsi_objek_utama": (round(prof["objects"][0]["n_postings"] / max(1, prof["n_postings"]), 2)
                              if prof["objects"] else 0.0),
        "contoh_frasa": prof["example_phrases"][:10],
        "contoh_kalimat": [s[:240] for s in prof.get("example_sentences", [])[:5]],
    }
    return [{"role": "system", "content": MARK_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def mark_family(prof: dict, mark: Callable[[list[dict]], str | None], attempts: int = 2) -> dict:
    """The marks of one competency; asked again once when the answer cannot be read.
    An unreadable answer marks nothing, and says so."""
    problem = ""
    for n in range(1, attempts + 1):
        raw = mark(mark_messages(prof))
        try:
            a = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip()))
        except (TypeError, ValueError):
            problem = "the answer is not JSON"
            continue
        if not (isinstance(a, dict) and isinstance(a.get("bukan_pekerjaan"), bool)
                and isinstance(a.get("keluarga_campuran"), bool)):
            problem = "bukan_pekerjaan and keluarga_campuran must be true or false"
            continue
        return {"bukan_pekerjaan": a["bukan_pekerjaan"], "keluarga_campuran": a["keluarga_campuran"],
                "alasan": str(a.get("alasan") or "").strip()[:300], "percobaan": n}
    return {"bukan_pekerjaan": False, "keluarga_campuran": False, "alasan": "",
            "masalah": problem, "percobaan": attempts}


def mark_flags(marks: dict | None) -> list[str]:
    if not marks:
        return []
    reason = marks.get("alasan") or "ditandai pemeriksa"
    return ([f"bukan_pekerjaan: {reason}"] if marks.get("bukan_pekerjaan") else []) + \
           ([f"keluarga_campuran: {reason}"] if marks.get("keluarga_campuran") else [])


def repeated_sentences(rows: Sequence[dict], cfg: FormulateConfig) -> str | None:
    """A mark when the competency's postings share very few distinct sentences:
    the evidence is then one employer's text repeated (a company description, a
    benefit, a call to apply), not a skill many employers ask for."""
    posts = {r.get("posting_id") for r in rows}
    sentences = {_norm(r.get("sentence_text") or "") for r in rows}
    if len(posts) < cfg.repeated_sentence_min_postings:
        return None
    if len(sentences) / len(posts) > cfg.repeated_sentence_ratio:
        return None
    return (f"kalimat_berulang: {len(posts)} lowongan memakai hanya {len(sentences)} kalimat "
            f"berbeda (kemungkinan teks pemberi kerja, bukan keterampilan)")


def formulate_one(c: dict, rows: Sequence[dict], index: SkkniIndex,
                  call: Callable[[list[dict]], str | None], cfg: FormulateConfig | None = None,
                  confirm: Callable[[list[dict]], str | None] | None = None,
                  mark: Callable[[list[dict]], str | None] | None = None) -> dict:
    """One competency: mark it with `mark` (a second model), write, check (one
    retry), decide the wording, and confirm a 'sama' with `confirm`. Without
    `confirm` no 'sama' is confirmed, so no SKKNI wording is used; without `mark`
    nothing is marked but the free 'kalimat_berulang'."""
    cfg = cfg or FormulateConfig()
    if is_fragment(c["statement"]["text"]):
        # a phrase cut before its object names no work: nothing to formulate, no call
        return {"competency_id": c["competency_id"], "version": FORMULATE_VERSION,
                "accepted": False, "skipped": True,
                "level": c["cognitive"].get("level"), "code": c["cognitive"].get("code"),
                "n_postings": c["n_postings"], "pernyataan_aturan": c["statement"]["text"],
                "problems": ["fragment: the phrases were cut before their object (extraction error)"],
                "flags": ["fragmen"], "attempts": []}
    prof = profile(c, rows)
    chosen = choose_qualifiers(prof, cfg)
    cands = index.nearest(competency_text(prof), cfg.n_candidates)
    obj_words = [w.lower() for w in (c["statement"].get("object") or "").split()]
    flags = []
    if obj_words and obj_words[-1] in GENERIC_OBJECTS:
        flags.append("terlalu_umum: objeknya umum ('" + " ".join(obj_words) + "')"
                     + ("" if chosen else " dan lowongan tidak menyebut standar, mutu, atau hasil"))
    repeated = repeated_sentences(rows, cfg)
    if repeated:
        flags.append(repeated)
    marks = mark_family(prof, mark) if mark else None
    flags += mark_flags(marks)
    attempts, problems, reserve = [], [], None
    for i in range(2):
        raw = call(build_messages(prof, chosen, cands, problems))
        a, problems = check(raw, prof, chosen, cands)
        slips = a.pop("_skkni_slips", []) if a is not None else []
        if a is not None and slips and i == 0:
            # a good statement with a slip in its SKKNI fields: ask once more for a
            # consistent answer, and keep this one (its link read cautiously) in case
            # the second answer is worse
            reserve, problems = a, list(slips)
            attempts.append({"problems": problems, "disimpan": True})
            continue
        attempts.append({"problems": problems})
        if a is None and reserve is not None:
            a = reserve
            attempts[-1]["dipakai_jawaban"] = 1
        if a is not None:
            w = decide_wording(a, prof, cands)
            if a["skkni"].get("catatan"):
                w["catatan_skkni"] = a["skkni"]["catatan"]
            if marks and marks["bukan_pekerjaan"] and w["skkni"]:
                # what is not work has no unit of work: no link, and no confirmation call
                w["catatan_skkni"] = w.get("catatan_skkni", []) + [
                    f"ditandai bukan pekerjaan: tautan {w['skkni']['kode']} "
                    f"({w['skkni']['hubungan']}) tidak dipakai"]
                w["skkni"] = None
                if w["sumber_rumusan"].startswith("SKKNI"):
                    w["pernyataan"] = w["pernyataan_pasar"]
                    w["statement_en"] = w.pop("statement_en_pasar", w["statement_en"])
                    w["sumber_rumusan"] = "pasar"
            if w["skkni"] and w["skkni"]["hubungan"] == "sama":
                unit = next(u for u, _ in cands if u["code"] == w["skkni"]["kode"])
                w = apply_confirmation(w, confirm_same(prof, unit, confirm) if confirm else None)
            return {"competency_id": c["competency_id"], "version": FORMULATE_VERSION,
                    "accepted": True,
                    "level": prof.get("level"), "code": prof.get("code"),
                    "n_postings": prof["n_postings"], "pernyataan_aturan": prof["statement"],
                    **w, "tanda": marks,
                    "alasan_tanpa_kualifikasi": str(a.get("alasan_tanpa_kualifikasi") or "").strip(),
                    "kualifikasi": [{k: q.get(k) for k in ("id", "kind", "text", "n_postings",
                                                            "share", "iso25010") if k in q}
                                    for q in chosen if q["id"] in set(a.get("kualifikasi_dipakai") or [])],
                    "kualifikasi_tersedia": [q["id"] for q in chosen],
                    "kandidat_skkni": [{"kode": u["code"], "judul": u["title"], "kemiripan": s}
                                       for u, s in cands],
                    "flags": flags, "attempts": attempts}
    return {"competency_id": c["competency_id"], "version": FORMULATE_VERSION,
            "accepted": False, "level": prof.get("level"),
            "code": prof.get("code"), "n_postings": prof["n_postings"],
            "pernyataan_aturan": prof["statement"], "problems": problems, "flags": flags,
            "tanda": marks, "attempts": attempts}


def pick_sample(comps: Sequence[dict], n: int, seed: int, status: str | None) -> list[dict]:
    """A sample spread over the cognitive levels, most demanded first within a
    level, with a seeded shuffle among equals."""
    pool = [c for c in comps if (status is None or c["contract"]["status"] == status)
            and not is_fragment(c["statement"]["text"])]
    by_level: dict[str, list[dict]] = {}
    for c in pool:
        by_level.setdefault(c["cognitive"].get("code") or "none", []).append(c)
    rng = random.Random(seed)
    for v in by_level.values():
        rng.shuffle(v)
        v.sort(key=lambda c: -c["n_postings"] // 50)       # coarse: frequent first, then random
    out, keys = [], sorted(by_level)
    while len(out) < n and any(by_level.values()):
        for k in keys:
            if by_level[k] and len(out) < n:
                out.append(by_level[k].pop(0))
    return out


def main(argv=None) -> int:
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None, help="a sample spread over levels")
    ap.add_argument("--all", action="store_true", help="every competency of --status")
    ap.add_argument("--status", default="kandidat_tinjauan", help="'' for every status")
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out", default="formulated")
    ap.add_argument("--confirm-model", default=CONFIRM_MODEL,
                    help="the second model that must agree before SKKNI wording is used")
    ap.add_argument("--confirm-max-usd", type=float, default=0.10)
    ap.add_argument("--mark-model", default=MARK_MODEL,
                    help="the model that marks competencies that are not work or mix works")
    ap.add_argument("--mark-max-usd", type=float, default=0.30)
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave out competencies the output file already holds (no second payment)")
    args = ap.parse_args(argv)
    from fivetag.llm import capped_caller
    from labeling.run import DEFAULT_MODELS, BudgetError
    from .endtoend import STATEMENT_MODEL
    from .qualifiers import load_run
    from .skkni import load_units
    run_dir = Path(args.run)
    comps, rows = load_run(run_dir)
    by_id = {c["competency_id"]: c for c in comps}
    status = args.status or None
    if args.ids:
        chosen = [by_id[i] for i in args.ids]
    elif args.all:
        chosen = [c for c in comps if status is None or c["contract"]["status"] == status]
    else:
        chosen = pick_sample(comps, args.sample or 20, args.seed, status)
    units = load_units()
    index = SkkniIndex(units)
    cfg = FormulateConfig()
    out_path = run_dir / f"{args.out}.json"
    store = (json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file()
             else {"statements": {}})
    if args.skip_existing:
        before = len(chosen)
        chosen = [c for c in chosen if c["competency_id"] not in store["statements"]]
        print(f"skipping {before - len(chosen)} competencies already in {out_path.name}")
    call = capped_caller(STATEMENT_MODEL, DEFAULT_MODELS[STATEMENT_MODEL], args.max_usd,
                         max_tokens=cfg.max_tokens, log_path=run_dir / f"{args.out}_llm_log.jsonl")
    # room for hidden reasoning before the short JSON answer
    confirm = capped_caller(args.confirm_model, DEFAULT_MODELS[args.confirm_model],
                            args.confirm_max_usd, max_tokens=cfg.confirm_max_tokens,
                            log_path=run_dir / f"{args.out}_confirm_log.jsonl")
    marker = capped_caller(args.mark_model, DEFAULT_MODELS[args.mark_model],
                           args.mark_max_usd, max_tokens=cfg.confirm_max_tokens,
                           log_path=run_dir / f"{args.out}_mark_log.jsonl")
    n_done = 0
    for c in chosen:
        try:
            r = formulate_one(c, rows[c["competency_id"]], index, call, cfg, confirm=confirm,
                              mark=marker)
        except BudgetError as exc:
            print(f"stopped: {exc}")
            break
        store["statements"][c["competency_id"]] = r
        n_done += 1
        s = r.get("skkni") or {}
        t = r.get("tanda") or {}
        print(f"{'OK ' if r['accepted'] else 'REJ'} {r.get('code') or '--':4s} "
              f"{(r.get('pernyataan') or '; '.join(r.get('problems', [])))[:90]}"
              + (f"  [{s.get('hubungan')} {s.get('kode')}"
                 + ("" if "dikonfirmasi" not in s else
                    " confirmed" if s["dikonfirmasi"] else
                    f" NOT confirmed: second model {(s.get('konfirmasi') or {}).get('hubungan')}")
                 + "]" if s else "")
              + (" {NOT WORK}" if t.get("bukan_pekerjaan") else "")
              + (" {MIXED}" if t.get("keluarga_campuran") else "")
              + (f" {{mark unreadable: {t['masalah']}}}" if t.get("masalah") else ""))
    vals = list(store["statements"].values())
    this_call = {"version": FORMULATE_VERSION, "n": n_done,
                 "spent_usd_estimate": round(call.spent(), 6), "n_calls": call.n_calls,
                 "confirm_spent_usd_estimate": round(confirm.spent(), 6),
                 "confirm_n_calls": confirm.n_calls,
                 "mark_spent_usd_estimate": round(marker.spent(), 6), "mark_n_calls": marker.n_calls}
    old = store.get("info") or {}
    # earlier notes on the file (its samples) are kept; every run into it is listed
    store["info"] = {**old, "formulate_version": FORMULATE_VERSION, "settings": asdict(cfg),
                     "model": STATEMENT_MODEL, "confirm_model": args.confirm_model,
                     "mark_model": args.mark_model,
                     "n_skkni_units": len(units),
                     "skkni_documents": sorted({u.get("source", "") for u in units}),
                     "n_statements": len(vals), "n_accepted": sum(v["accepted"] for v in vals),
                     "this_call": this_call, "calls": list(old.get("calls", [])) + [this_call]}
    out_path.write_text(json.dumps(store, indent=1, ensure_ascii=False), encoding="utf-8")
    write_csv(vals, run_dir / f"{args.out}.csv")
    print(json.dumps(store["info"]["this_call"]), f"-> {out_path}")
    return 0


def write_csv(vals: Sequence[dict], path: Path) -> None:
    """The statements as a table for Excel, most demanded first."""
    import csv
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["competency_id", "level", "n_postings", "pernyataan", "statement_en",
                    "sumber_rumusan", "skkni_kode", "skkni_judul", "skkni_tahun", "hubungan",
                    "dikonfirmasi", "tingkat_judul_skkni", "kualifikasi", "flags",
                    "pernyataan_aturan"])
        for v in sorted(vals, key=lambda v: -v["n_postings"]):
            s = v.get("skkni") or {}
            w.writerow([v["competency_id"], v.get("code") or "", v["n_postings"],
                        v.get("pernyataan") or "", v.get("statement_en") or "",
                        v.get("sumber_rumusan") or "DITOLAK", s.get("kode") or "",
                        s.get("judul") or "", s.get("tahun") or "", s.get("hubungan") or "",
                        "" if "dikonfirmasi" not in s else ("ya" if s["dikonfirmasi"] else "tidak"),
                        s.get("kode_tingkat_judul") or "",
                        "; ".join(f"{q['text']} ({q['n_postings']})" for q in v.get("kualifikasi", [])),
                        "; ".join(v.get("flags", [])), v["pernyataan_aturan"]])


if __name__ == "__main__":
    raise SystemExit(main())
