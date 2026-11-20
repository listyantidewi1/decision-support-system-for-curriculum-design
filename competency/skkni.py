"""SKKNI competency units, read in full from the official documents.

`DATA/master_skkni/json/skkni.json` holds unit TITLES only. The PDFs in
`DATA/master_skkni/` hold the whole units: for each, the description, the
elements of competency with their performance criteria (kriteria unjuk kerja,
KUK), the range of variables (context, tools and equipment, regulations, norms
and standards) and the assessment guide (knowledge, skills, work attitude,
critical aspects). This module parses that text.

Why it matters (the promotors' definition, 2026-11-20): a competency is not a
task title such as "develop web applications"; it says how well the work is
done, which standards apply and what observable result it gives. An SKKNI unit
does exactly that through its elements and KUK. Parsed units serve the
competency layer twice: as the FORMAT a market-derived unit is written in, and
as the REFERENCE it is compared with.

Text comes from the PDF with pypdf, its pages separated by a form-feed line.
Two-column tables come out cell by cell, which this parser relies on; words
broken by the PDF ("metod ologi") are left as they are. Every parsed unit is
checked: a unit with no element or with an element without KUK is reported, not
silently kept.

One standard has another layout: it adopts units of an international ICT
training package (codes such as ICTWEB416, ICAICT201A) and prints every unit in
English and in Indonesian. `parse_training_package` reads it (see the comment
above it); `parse_all` uses it for a document that has training-package units
and no "KODE UNIT : J." unit.

    python -m competency.skkni parse "DATA/master_skkni/Software Development Pemrograman.pdf"
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKKNI_DIR = REPO_ROOT / "DATA" / "master_skkni"
PARSER_VERSION = "skkni-parser-v2"  # v2: adopted training-package units (2019 standard)

# codes seen: J.620100.001.01, J.63SAM00.001.2, J.62DMS00.001.1, J.61IOT01.001.1
# one code is misprinted with five digits in its document (J.62010.033.02)
_UNIT = re.compile(r"KODE UNIT\s*:\s*(J\.[0-9A-Z]{5,8}\.\d{3}\.\d{1,2})")
_DECREE = re.compile(r"KEPUTUSAN\s+MENTERI[\s\S]{0,160}?NOMOR\s+(\d+)\s+TAHUN\s+(\d{4})",
                     re.IGNORECASE)
_PAGE = re.compile(r"^\s*\d{1,3}\s*$")
_ELEMENT = re.compile(r"^(\d{1,2})\.\s+(\S.*)$")
_CRITERION = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+(\S.*)$")
_SUB = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+(\S.*)$")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _between(text: str, start: str, ends: list[str]) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = min([k for k in (text.find(e, i) for e in ends) if k >= 0] or [len(text)])
    return text[i:j]


def _numbered(block: str) -> list[tuple[str, str]]:
    """'3.1.1 Something that runs over two lines' -> [('3.1.1', 'Something ...')]."""
    out: list[list[str]] = []
    for line in block.splitlines():
        if _PAGE.match(line) or not line.strip():
            continue
        m = _SUB.match(line.strip()) or _ELEMENT.match(line.strip())
        if m and m.lastindex == 2:
            out.append([m.group(1), m.group(2)])
        elif out:
            out[-1][1] += " " + line.strip()
    return [(n, _clean(t)) for n, t in out]


def _elements(block: str) -> list[dict]:
    elements: list[dict] = []
    current: dict | None = None
    in_title = False
    for raw in block.splitlines():
        line = raw.strip()
        if not line or _PAGE.match(line) or line.startswith("ELEMEN KOMPETENSI"):
            continue
        m = _CRITERION.match(line)
        if m and current is not None and m.group(1) == current["number"]:
            current["kuk"].append({"number": f"{m.group(1)}.{m.group(2)}", "text": m.group(3)})
            in_title = False
            continue
        m = _ELEMENT.match(line)
        if m and (current is None or int(m.group(1)) == int(current["number"]) + 1):
            current = {"number": m.group(1), "title": m.group(2), "kuk": []}
            elements.append(current)
            in_title = True
            continue
        if current is None:
            continue
        if in_title:
            current["title"] += " " + line
        elif current["kuk"]:
            current["kuk"][-1]["text"] += " " + line
    for e in elements:
        e["title"] = _clean(e["title"])
        for k in e["kuk"]:
            k["text"] = _clean(k["text"])
    return elements


def _list_under(block: str, number: str) -> list[str]:
    """Items numbered `number.k` (or `number.k.l`) inside a block."""
    return [t for n, t in _numbered(block)
            if n.startswith(number + ".") and n.count(".") == number.count(".") + 1]


def parse_units(text: str, source: str = "") -> list[dict]:
    starts = [(m.start(), m.group(1)) for m in _UNIT.finditer(text)]
    units = []
    for (start, code), nxt in zip(starts, starts[1:] + [(len(text), "")]):
        body = text[start:nxt[0]]
        title = _clean(_between(body, "JUDUL UNIT", ["DESKRIPSI UNIT"]).lstrip(" :"))
        description = _clean(_between(body, "DESKRIPSI UNIT", ["ELEMEN KOMPETENSI"]).lstrip(" :"))
        elements = _elements(_between(body, "ELEMEN KOMPETENSI", ["BATASAN VARIABEL"]))
        ranges = _between(body, "BATASAN VARIABEL", ["PANDUAN PENILAIAN"])
        guide = _between(body, "PANDUAN PENILAIAN", ["KODE UNIT"])
        nums = dict(_numbered(guide))
        heads = {re.sub(r"\s+", " ", t).lower(): n for n, t in _numbered(guide) if n.count(".") == 0}

        def section(word: str) -> str | None:
            return next((n for t, n in heads.items() if word in t), None)
        know_skill = section("pengetahuan")
        attitude = section("sikap kerja")
        critical = section("aspek kritis")
        prereq = section("persyaratan")
        knowledge = skills = []
        if know_skill:
            sub = {t.lower(): n for n, t in _numbered(guide)
                   if n.startswith(know_skill + ".") and n.count(".") == 1}
            k_no = next((n for t, n in sub.items() if t.startswith("pengetahuan")), None)
            s_no = next((n for t, n in sub.items() if t.startswith("keterampilan")), None)
            knowledge = _list_under(guide, k_no) if k_no else []
            skills = _list_under(guide, s_no) if s_no else []
        rnums = _numbered(ranges)
        tools = [t for n, t in rnums if n.startswith("2.1.") and n.count(".") == 2]
        equipment = [t for n, t in rnums if n.startswith("2.2.") and n.count(".") == 2]
        standards = [t for n, t in rnums if n.startswith("4.2.") and n.count(".") == 2]
        norms = [t for n, t in rnums if n.startswith("4.1.") and n.count(".") == 2]
        context = [t for n, t in rnums if n.startswith("1.") and n.count(".") == 1]
        units.append({
            "code": code, "title": title, "description": description, "elements": elements,
            "range": {"context": context, "tools": tools, "equipment": equipment,
                      "norms": norms, "standards": standards},
            "knowledge": knowledge, "skills": skills,
            "attitudes": _list_under(guide, attitude) if attitude else [],
            "critical_aspects": _list_under(guide, critical) if critical else [],
            "prerequisites": [t for t in (_list_under(guide, prereq) if prereq else [])
                              if t.startswith("J.")],
            "source": source,
            "problems": ([] if elements else ["no element parsed"])
                        + [f"element {e['number']} has no KUK" for e in elements if not e["kuk"]],
        })
    return units


# ------------------------------------------------------- training-package units
# One standard in the folder (Kepmenaker 22/2019, "Programming and Software
# Development, Network and Infrastructure, Operation and System Tools") does not
# use the SKKNI layout. It adopts units of an international ICT training package
# (codes such as ICTWEB416, ICAICT201A, BSBDES401; a few have four digits,
# ICTTEN5201A) and prints every unit twice, in English and in Indonesian. What
# this parser relies on, read from that document's text:
#  - a unit opens with a cover page ("CODE English title / Indonesian title"),
#    then the English heading "CODE English title" followed within three lines
#    by "Modification History"; the Indonesian heading "CODE judul" is followed
#    by "Riwayat Perubahan" (or "Sejarah Perubahan");
#  - the two languages alternate page by page, often inside one section, and a
#    page that continues a table repeats no heading: the language of such a page
#    is read from its function words, and the page continues the section that
#    language was in;
#  - two layouts occur. The newer: Application, Elements and Performance
#    Criteria, Foundation Skills, then Assessment Requirements with Performance
#    Evidence and Knowledge Evidence. The older (codes ending in A, B or C):
#    Unit Descriptor, Application of the Unit, Pre-Requisites, Elements and
#    Performance Criteria, Required Skills and Knowledge, Evidence Guide (with
#    its critical aspects) and Range Statement;
#  - pypdf ends a line with a space where a word ended and without one where it
#    cut a word ("mengiden" + "tifikasi", "ide-" + "ide"); such lines are glued;
#  - a page starts with its number, but 26 pages have none: read_pdf separates
#    the pages with a form-feed line, and either marks a page break;
#  - a page often holds the untranslated English of the page before it with a
#    few words translated (first-aid terms, product names): a page that repeats
#    the lines of the page before is that page's twin in the other language,
#    whatever its function words say.
# The cover's code is the one the decree lists (BSBSUS201A); the headings of a
# few units print the newer code of the same unit (BSBSUS201). The Indonesian
# title is the one of the decree's list of units (Daftar Unit Kompetensi), the
# official wording; the unit's own heading is a rougher translation (31 of 79
# differ, "Merencanakan, melaksanakan dan solusi komunikasi tes perusahaan" for
# the list's "Merencanakan, Mengimplementasikan dan Menguji Solusi Komunikasi
# Perusahaan"). The list keeps its own typos ("Menyelilidiki").
# Mapping to the SKKNI schema: description = Indonesian descriptor and
# application text; skills = required skills (older layout) or performance
# evidence (newer layout); knowledge = required knowledge or knowledge
# evidence; critical_aspects = the evidence guide's critical aspects (older
# layout only); range = the range statement (older layout only); attitudes stay
# empty, no training-package unit has a work-attitude section.

_TP_CODE = r"[A-Z]{6}\d{3,4}[A-Z]?"
_TP_CODE_LINE = re.compile(rf"^\s*({_TP_CODE})\b[ \t]*(.*)$")
_TP_PAGE = re.compile(r"^\s*\d{1,4}\s*$")
_TP_EN_HEAD_END = re.compile(r"(?i)^modification history\b")
_TP_ID_HEAD_END = re.compile(r"(?i)^((riwayat|sejarah) (perubahan|perbaikan|modifikasi)|rilis komentar)\b")
# bullets: two private-use glyphs of the PDF's symbol font (U+F0B7, U+F0A7), the
# bullet U+2022, the black circle U+25CF, the small squares and circles U+25AA,
# U+25E6, and "o" for a sub-item. A bullet may be printed twice ("● •Pengelola
# jaringan"), and an "o" alone on a line is an empty sub-item.
_TP_GLYPHS = "".join(map(chr, (0xF0B7, 0xF0A7, 0x2022, 0x25CF, 0x25AA, 0x25E6)))
_TP_BULLET = re.compile(rf"^(?:[{_TP_GLYPHS}]|o(?=\s|$))(?:\s*[{_TP_GLYPHS}])*\s*")
_TP_INLINE_BULLET = re.compile(f"[{_TP_GLYPHS[:4]}]")
# a text that stops at a word which cannot end a phrase: a conjunction, a
# preposition, an article, or "yang" + an adverb ("di browser yang sering"). Case
# matters: "Vitamin A" ends a phrase.
_TP_OPEN_END = re.compile(
    r"(?<![\w-])(dan|atau|serta|yang|untuk|dari|di|ke|dengan|pada|oleh|bagi|terhadap|tentang"
    r"|sebagai|seperti|antara|and|or|of|to|for|the|a|an|with|from|into|that|which|than"
    r"|yang (sering|paling|sangat|lebih|telah|sudah|akan|sedang|masih|belum|tidak|dapat|bisa"
    r"|harus|perlu|mungkin))$")
# "1.1 Text", "1.1. Teks", "2.3 .Teks", "5..4 Teks" (misprints of the document) and
# "2,5 Teks" (the translation's decimal comma)
_TP_KUK = re.compile(r"^(\d{1,2})(?:\s*\.{1,2}\s*|,)(\d{1,2})(?!\d)\s*\.?\s*(.*)$")
# "1 Define", "1. Menentukan", "1.Berkomunikasi", or the number alone
_TP_ELEMENT = re.compile(r"^(\d{1,2})(?:\.\s*|\s+|$)(\D.*)?$")
_TP_NOTE = re.compile(r"(?i)^(note|catatan)\s*:")
# the explanation at the head of an element table ("Elements describe the essential
# outcomes ...", "Kriteria unjuk kerja menggambarkan ..."), printed again on a later
# page of some tables
_TP_EXPLANATION = re.compile(r"(?i)^(elements? (describe|define)|elemen (menggambarkan|menentukan|menjelaskan)"
                             r"|performance criteria (describe|specify)"
                             r"|kriteria (unjuk kerja|kinerja) (menggambarkan|menentukan|menjelaskan))\b")

# Section headings, matched against the whole line (lower case, spaces collapsed,
# a final ":" dropped) -> (language, section). "other" closes the section before
# it. Three Indonesian headings mean different things in different places
# (_tp_heading decides): "skills?" (Keterampilan yang dibutuhkan) is the "required
# skills" sub-heading inside the older layout's Required Skills and Knowledge,
# the performance evidence in the Assessment Requirements part, and Foundation
# Skills elsewhere; "knowledge?" (Pengetahuan yang dibutuhkan) is a sub-heading
# or the knowledge evidence; "evidence?" (Bukti Penilaian) is the performance
# evidence at the start of the Assessment Requirements part and the assessment
# conditions after its knowledge evidence.
_TP_HEADINGS = [(re.compile(p), lang, name) for p, lang, name in [
    (r"modification history|(release|version) comments", "en", "history"),
    (r"unit descriptor", "en", "descriptor"),
    (r"application( of the unit)?", "en", "application"),
    (r"pre-?requisites?( units?)?", "en", "prerequisites"),
    (r"elements and performance criteria pre-content", "en", "other"),
    (r"elements? (and )?performance criteria", "en", "elements"),
    (r"foundation skills", "en", "foundation"),
    (r"required skills and knowledge", "en", "required"),
    (r"evidence guide", "en", "evidence"),
    (r"range (statement|of conditions)", "en", "range"),
    (r"performance evidence", "en", "performance"),
    (r"knowledge evidence", "en", "knowledge"),
    (r"assessment requirements for .*", "en", "assessment"),
    (r"licensing/regulatory information|employability skills information|assessment conditions"
     r"|unit mapping information|links|unit sectors?( ?\(s\))?|competency field"
     r"|co-?requisite units?|assessor requirements", "en", "other"),
    (r"(riwayat|sejarah) (perubahan|perbaikan|modifikasi)|(rilis|versi) (komentar|keterangan)",
     "id", "history"),
    (r"(deskripsi|penjelasan) unit", "id", "descriptor"),
    (r"penerapan( unit)?", "id", "application"),
    (r"(unit-unit |unit |satuan )?pra-?syarat( unit)?", "id", "prerequisites"),
    (r"(elemen|elemen-elemen|unsur-unsur) dan kriteria (unjuk kerja|unjuk kinerja|kinerja)"
     r" ?- ?penjelasan", "id", "other"),
    (r"(elemen|elemen-elemen|elemen kompetensi|unsur-unsur)( dan)? kriteria "
     r"(unjuk kerja|unjuk kinerja|kinerja)", "id", "elements"),
    (r"(keterampilan|ketrampilan) dasar", "id", "foundation"),
    (r"(keterampilan|ketrampilan|kecakapan) yang (dibutuhkan|diperlukan)", "id", "skills?"),
    (r"pengetahuan yang (dibutuhkan|diperlukan)", "id", "knowledge?"),
    (r"bukti penilaian", "id", "evidence?"),
    (r"(keterampilan|ketrampilan|kecakapan) dan pengetahuan yang (dibutuhkan|diperlukan)",
     "id", "required"),
    (r"panduan (penilaian|asesmen|bukti)", "id", "evidence"),
    (r"batasan variabel", "id", "range"),
    (r"(bukti|penilaian) (unjuk kerja|kinerja)", "id", "performance"),
    (r"(bukti|penilaian) pengetahuan", "id", "knowledge"),
    (r"persyaratan penilaian untuk .*", "id", "assessment"),
    (r"informasi (sertifikasi|lisensi|perizinan).*|informasi kecakapan kerja"
     r"|(ketentuan|kondisi) penilaian|informa?si pemetaan unit|tautan|sektor unit( ?\(s\))?"
     r"|unit sektor|bidang kompetensi|unit terkait|persyaratan asesor", "id", "other"),
]]
# the descriptor and application headings count only at the top of a unit: the
# word "Application" also opens a range variable ("Application software may include")
_TP_OPENING = (None, "title", "history", "descriptor", "application")

# function words that tell the two languages apart on a page without a heading
_TP_EN_WORDS = frozenset("the and of to in for with on by is are be that from as this "
                         "must may include including an a".split())
_TP_ID_WORDS = frozenset("dan yang untuk dengan di ke dari atau dalam pada ini itu oleh "
                         "secara akan tidak adalah sebagai tentang sesuai harus dapat "
                         "termasuk mungkin".split())

# sub-headings of the older layout's Required Skills and Knowledge
_TP_REQ_SKILLS = re.compile(r"(?i)^(required skills|(keterampilan|ketrampilan|kemampuan|kecakapan) yang "
                            r"(dibutuhkan|diperlukan)|(dibutuhkan|diperlukan) (keterampilan|ketrampilan|kecakapan))$")
_TP_REQ_KNOWLEDGE = re.compile(r"(?i)^(required knowledge|pengetahuan yang (dibutuhkan|diperlukan)"
                               r"|(dibutuhkan|diperlukan) pengetahuan)$")
# labels of the older layout's evidence guide: where the critical aspects start and
# end (a label may be printed with a bullet, as if it were an aspect)
_TP_CRITICAL = re.compile(r"(?i)^(critical aspects?|aspek(-aspek)?)\b")
_TP_GUIDE_LABEL = re.compile(r"(?i)^(context of|konteks dan|method of|metode( penilaian| asesmen)?$"
                             r"|metode (penilaian|asesmen)\b|guidance|informasi panduan|panduan informasi"
                             r"|panduan untuk)")
# the row label in front of a descriptor or application text: "Deskripsi unit Unit
# ini ...", "Deskripsi Umit Unit ini ...", "Deskripsi Unit ini ..." (keeps "Unit ini")
_TP_LABEL = re.compile(r"(?i)^(unit descriptor|application of the unit"
                       r"|(deskripsi|penjelasan|aplikasi|penerapan) (unit|umit)(?! ini\b)"
                       r"|(deskripsi|penjelasan|aplikasi|penerapan)(?= unit ini\b))\s*:?\s*")
# a range variable whose items are tools, equipment, norms or standards; any other
# variable stays in "context" as one line (variable and its items)
# A variable is a tool or equipment only when the artefact word names what it is,
# not what it is about: Indonesian puts the head noun first ("Situs aplikasi
# internet" are sites, "metodologi pengembangan perangkat lunak" methods), English
# last ("Internet application sites may include").
_TP_NOT_ARTEFACT = re.compile(r"(?i)^(yang termasuk\s+)?(situs|metodologi|metode)\b"
                              r"|\b(sites|methodolog(y|ies)|methods)\s+(may|dapat|mungkin|termasuk|include)\b")
_TP_RANGE_LISTS = [
    ("standards", re.compile(r"(?i)standar")),
    ("norms", re.compile(r"(?i)legisla|peraturan|regulat|undang|hukum|\blaw|kode etik|code of"
                         r"|etika|ethic|norma")),
    ("tools", re.compile(r"(?i)perangkat lunak|software|aplikasi|application|\btools?\b|\balat\b"
                         r"|utilit")),
    ("equipment", re.compile(r"(?i)perangkat keras|hardware|peralatan|equipment|perlengkapan"
                             r"|komponen|component|device|perangkat")),
]


def _tp_base(code: str) -> str:
    """ICADBS503B -> ICADBS503: the same unit under its older and newer code."""
    return re.sub(r"[A-Z]$", "", code)


def _tp_is_page(lines: list[str], i: int) -> bool:
    """A page number: a line that is only a number, after an empty line (a year
    alone on a line inside a list follows text, not an empty line)."""
    return bool(_TP_PAGE.match(lines[i])) and (i == 0 or not lines[i - 1].strip())


def _tp_is_feed(line: str) -> bool:
    """The form-feed line read_pdf writes between two pages."""
    return "\f" in line and not line.strip()


def _tp_is_break(lines: list[str], i: int) -> bool:
    """A page break: the form-feed line between two pages, or a page number."""
    return _tp_is_feed(lines[i]) or _tp_is_page(lines, i)


def _tp_unfinished(text: str) -> bool:
    """A title, criterion or item that breaks off ("... untuk pemikiran dan")."""
    return bool(_TP_OPEN_END.search(text.rstrip()))


def _tp_norm(line: str) -> str:
    """A line as compared across pages: no bullet, lower case, single spaces, no
    final stop."""
    return re.sub(r"\s+", " ", _TP_BULLET.sub("", line.strip())).lower().rstrip(" .;,:")


def _tp_twin(chunk: list[str], before: dict[str, tuple]) -> tuple | None:
    """When the lines at the top of a page repeat lines of the page before it (an
    untranslated, or partly translated, copy of that page), the (language,
    section) those lines were read in; else None. `before` maps each line of the
    page before (_tp_norm) to its (language, section). Lines of one word
    ("diabetes", "Criteria") do not count, table headers repeat them."""
    mine = [n for n in dict.fromkeys(_tp_norm(line) for line in chunk) if len(n.split()) >= 2]
    shared = [before[n] for n in mine if n in before]
    if len(shared) >= 3 and len(shared) >= 0.3 * len(mine):
        return Counter(shared).most_common(1)[0][0]
    return None


def _tp_join(lines: list[str]) -> str:
    """Lines -> one string. A line that pypdf ended inside a word (no space at its
    end) is glued to the next one; an empty line separates."""
    out, glue = "", False
    for raw in lines:
        piece = raw.strip()
        if not piece:
            glue = False
            continue
        if not out:
            out = piece
        elif glue and not raw[:1].isspace():
            out += piece
        else:
            out += " " + piece
        glue = not raw[-1:].isspace()
    return _clean(out)


def _tp_follows(lines: list[str], i: int, rx: re.Pattern, n: int = 3) -> bool:
    """Does one of the n non-empty lines after line i match rx?"""
    seen = 0
    for line in lines[i + 1:]:
        if line.strip():
            if rx.match(line.strip()):
                return True
            seen += 1
            if seen == n:
                return False
    return False


def _tp_anchors(lines: list[str]) -> list[tuple[int, str]]:
    """(line, code) of every English unit heading: a code that starts a line,
    with "Modification History" within the next three lines."""
    out = []
    for i, line in enumerate(lines):
        m = _TP_CODE_LINE.match(line)
        if m and _tp_follows(lines, i, _TP_EN_HEAD_END):
            out.append((i, m.group(1)))
    return out


def _tp_cover(lines: list[str], lo: int, hi: int, base: str) -> int | None:
    """The cover line of a unit: the first line in the 60 before its English
    heading (and not before `lo`) that starts with the unit's code."""
    for i in range(max(lo, hi - 60), hi):
        m = _TP_CODE_LINE.match(lines[i])
        if m and _tp_base(m.group(1)) == base:
            return i
    return None


def _tp_heading(lines: list[str], i: int, base: str) -> tuple | None:
    """(language, section, text after the heading) when line i reads as a heading.
    The section may still depend on where the heading stands (_tp_place)."""
    m = _TP_CODE_LINE.match(lines[i])
    if m and _tp_base(m.group(1)) == base:          # the unit's own title heading
        if _tp_follows(lines, i, _TP_EN_HEAD_END):
            return "en", "title", m.group(2)
        if _tp_follows(lines, i, _TP_ID_HEAD_END):
            return "id", "title", m.group(2)
    # a heading starts with a capital: "keterampilan dan pengetahuan yang diperlukan"
    # or "links" (of "Configure WAN / links") in lower case are text that happens
    # to read like a heading
    if not lines[i].strip()[:1].isupper():
        return None
    key = re.sub(r"\s+", " ", lines[i]).strip().lower().rstrip(":").strip()
    if len(key) > 120:
        return None
    for rx, lang, name in _TP_HEADINGS:
        if rx.fullmatch(key):
            return lang, name, ""
    return None


def _tp_place(name: str, where: str | None) -> str | None:
    """The section a heading opens, given the section its language is in now;
    None when the line is text there, not a heading."""
    if name in ("skills?", "knowledge?") and where == "required":
        return None                               # a sub-heading of Required Skills and Knowledge
    if name == "skills?":
        return "performance" if where == "assessment" else "foundation"
    if name == "knowledge?":
        return "knowledge"
    if name == "evidence?":
        return "performance" if where == "assessment" else "other"
    if name == "history" and where == "assessment":
        return "assessment"                       # the history table of the Assessment Requirements
    if name in ("descriptor", "application") and where not in _TP_OPENING:
        return None
    return name


def _tp_language(lines: list[str]) -> str | None:
    """'en' or 'id' when the function words of the text clearly say so."""
    words = re.findall(r"[a-z]+", " ".join(lines).lower())
    en = sum(w in _TP_EN_WORDS for w in words)
    idn = sum(w in _TP_ID_WORDS for w in words)
    if en >= 2 and en > 2 * idn:
        return "en"
    if idn >= 2 and idn > 2 * en:
        return "id"
    return None


def _tp_chunk(body: list[str], i: int, base: str) -> list[str]:
    """Lines from line i up to the next heading or page break (at most 60)."""
    chunk = []
    for j in range(i, min(i + 60, len(body))):
        if _tp_is_break(body, j) or (body[j].strip() and _tp_heading(body, j, base)):
            break
        chunk.append(body[j])
    return chunk


# the opening sentence of every knowledge evidence list; it decides the section
# where the translation put the list under another heading ("Ketentuan Penilaian")
_TP_KNOWLEDGE_INTRO = re.compile(r"(?i)^(untuk melengkapi persyaratan unit|to complete the unit requirements)")


_TP_OTHER = {"en": "id", "id": "en"}
_TP_NO_CONTENT = (None, "title", "history", "other", "assessment")   # sections whose text is not kept
# a heading of one table row (the unit's sector, its competency field, related
# units), which never runs over a page; the translator sometimes prints it on the
# English page, before the end of the Indonesian section it follows
_TP_ONE_ROW = re.compile(r"(?i)^(unit sectors?( ?\(s\))?|competency field|co-?requisite units?"
                         r"|sektor unit( ?\(s\))?|unit sektor|bidang kompetensi|unit terkait)\s*:?$")


def _tp_page_language(body: list[str], i: int, base: str) -> str | None:
    """The language of the first heading further down the page of line i (as
    _tp_sections reads a heading's language), or None when there is none."""
    for j in range(i + 1, len(body)):
        if _tp_is_break(body, j):
            return None
        head = _tp_heading(body, j, base) if body[j].strip() else None
        if head is not None:
            return _tp_language(_tp_chunk(body, j + 1, base)) or head[0]
    return None


def _tp_sections(body: list[str], base: str) -> dict[tuple[str, str], list[str]]:
    """The raw lines of one unit sorted into (language, section). A heading names
    the section and, unless the text under it clearly reads as the other language
    (an English table header can head a translated page), the language. A page
    that starts without a heading is
      - the twin of the page before it when it repeats that page's lines
        (_tp_twin): the other language, same section;
      - else in the language its function words say; when they say nothing (a
        list of names, "post-it notes, reports"), in the language of the first
        heading further down the page; else in the language being read. It
        continues the section that language was last in, or, when that was a
        one-row section ("Sektor Unit"), the section before that row.
    Page numbers are dropped; empty lines are kept, they separate paragraphs for
    _tp_join, and a page break is kept as a form-feed line in front of the first
    line read after it (an item or a title cell may go on over it)."""
    out: dict[tuple[str, str], list[str]] = {}
    section: dict[str, str | None] = {"en": None, "id": None}
    resume: dict[str, str | None] = {"en": None, "id": None}   # the section before a one-row section
    lang, page_start, mark = "en", False, False
    page: dict[str, tuple] = {}                   # line (_tp_norm) -> (language, section), this page
    before: dict[str, tuple] = {}                 # ... and the page before it
    for i, raw in enumerate(body):
        if _tp_is_break(body, i):
            if not page_start:                    # the number after a form feed is the same break
                before, page = page, {}
            page_start = mark = True
            continue
        if raw.strip():
            head = _tp_heading(body, i, base)
            if head is not None and head[1] != "title":
                below = _tp_chunk(body, i + 1, base)
                side = _tp_language(below) or head[0]
                name = _tp_place(head[1], section[side])
                first = next((_TP_BULLET.sub("", l.strip()) for l in below if l.strip()), "")
                if name and _TP_KNOWLEDGE_INTRO.match(first):
                    name = "knowledge"
                head = (side, name, "") if name else None
            if page_start and head is None:           # the line may read like a heading
                chunk = [raw] + _tp_chunk(body, i + 1, base)
                twin = _tp_twin(chunk, before)
                if twin is not None:
                    lang = _TP_OTHER[twin[0]]
                    if twin[1] not in (None, "title"):
                        section[lang] = twin[1]
                else:
                    lang = _tp_language(chunk) or _tp_page_language(body, i, base) or lang
                    if section[lang] == "other" and resume[lang] is not None:
                        section[lang] = resume[lang]
                resume[lang] = None
            page_start = False
            if head is not None:
                lang, name, raw = head
                if name == "other" and _TP_ONE_ROW.match(body[i].strip()):
                    if section[lang] not in _TP_NO_CONTENT:
                        resume[lang] = section[lang]
                else:
                    resume[lang] = None
                section[lang] = name
                if not raw.strip():
                    continue
        if section[lang] is not None:
            lines = out.setdefault((lang, section[lang]), [])
            if raw.strip():
                if mark:
                    lines.append("\f")
                    mark = False
                page.setdefault(_tp_norm(raw), (lang, section[lang]))
            lines.append(raw)
    return out


def _tp_elements(lines: list[str]) -> list[dict]:
    """Elements and KUK of one language's element table. Text before element 1
    (the table's explanation) is skipped; an element is taken only as the next
    number, a criterion only under its own element; any other line continues the
    title or criterion being read. A table row cut by a page break goes on with
    its left cell: when the element title broke off ("4. Kembangkan kasus bisnis
    untuk"), the first block of lines after the break ends the title, and the
    lines after it go on with the criterion."""
    elements: list[dict] = []
    target: list[str] | None = None               # raw pieces of the text being read
    title_tail: list[str] | None = None           # an unfinished title going on after a page break
    tail_read = False                             # ... and whether a line of it was read
    for raw in lines:
        line = raw.strip()
        if _tp_is_feed(raw):
            if (elements and target is not None and target is not elements[-1]["parts"]
                    and _tp_unfinished(_tp_join(elements[-1]["parts"]))):
                title_tail, tail_read = elements[-1]["parts"], False
            continue
        if not line:
            if title_tail is not None and tail_read:
                title_tail = None                 # the title cell ends at the first empty line
            if target is not None:
                target.append("")
            continue
        tail = raw[len(raw.rstrip()):]            # keeps "no space at the end" for the glue
        current = int(elements[-1]["number"]) if elements else 0
        m = _TP_KUK.match(line)
        if m and int(m.group(1)) in (current, current + 1):
            title_tail = None
            if int(m.group(1)) == current + 1:    # a criterion whose element line was lost
                elements.append({"number": str(current + 1), "parts": [], "kuk": []})
            target = [m.group(3) + tail] if m.group(3) else []
            elements[-1]["kuk"].append({"number": f"{int(m.group(1))}.{int(m.group(2))}",
                                        "parts": target})
            continue
        m = _TP_ELEMENT.match(line)
        if m and int(m.group(1)) == current + 1:
            title_tail = None
            title, first = (m.group(2) or "") + tail, None
            # an element and its first criterion on one line: "5. Pilih solusi 5.1. Mengembangkan"
            s = re.search(rf"\s{m.group(1)}\s*\.\s*1(?!\d)\.?\s+", title)
            if s:
                title, first = title[:s.start()] + " ", title[s.end():]
            element = {"number": str(current + 1), "parts": [title], "kuk": []}
            elements.append(element)
            target = element["parts"]
            if first is not None:
                target = [first]
                element["kuk"].append({"number": f"{element['number']}.1", "parts": target})
            continue
        if title_tail is not None:
            title_tail.append(raw)
            tail_read = True
        elif target is not None:
            target.append(raw)
    return [{"number": e["number"], "title": _tp_join(e["parts"]),
             "kuk": [{"number": k["number"], "text": _tp_join(k["parts"])} for k in e["kuk"]]}
            for e in elements]


_TP_NUMBERED = re.compile(r"^(\d{1,2})\.\s+(\S.*)$")


def _tp_title_start(block: list[str]) -> int:
    """Where an element title begins among the lines between two criteria: at the
    first capitalised line (not glued to the line above) after the last line that
    closes a sentence; len(block) when there is none."""
    start = len(block)
    for i in range(len(block) - 1, -1, -1):
        s = block[i].strip()
        if not s:
            continue
        if s.endswith((".", ";")):
            break
        glued = i > 0 and block[i - 1].strip() != "" and not block[i - 1][-1:].isspace()
        if s[:1].isupper() and not glued:
            start = i
    return start


def _tp_flat_elements(lines: list[str]) -> list[dict]:
    """An element table numbered the older way (two screen-media units): the
    criteria are numbered 1, 2, 3 ... through the whole unit, and an element
    title carries no number (English) or its own number (the translation, where
    "1. Menetapkan prosedur" and "1. Mengkonfirmasi" both start with 1). Between
    two criteria, the lines before the title (_tp_title_start) end the first
    criterion. KUK are numbered per element as SKKNI numbers them: the fifth
    criterion of the unit, first of element 2, becomes 2.1."""
    numbers = []
    for raw in lines:
        m = _TP_NUMBERED.match(raw.strip())
        numbers.append(int(m.group(1)) if m else None)
    elements: list[dict] = []
    crit: list[str] | None = None                 # parts of the criterion being read
    title: list[str] | None = None                # parts of a numbered title being read
    pending: list[str] = []                       # unnumbered lines since the last criterion
    expected = 1                                  # number of the next criterion

    def criterion(text: str) -> None:
        nonlocal crit, title, pending, expected
        if title is not None:
            title.extend(pending)
        else:
            cut = _tp_title_start(pending)
            if crit is not None:
                crit.extend(pending[:cut])
            if pending[cut:] or not elements:
                elements.append({"parts": pending[cut:], "kuk": []})
        crit, title, pending = [text], None, []
        elements[-1]["kuk"].append(crit)
        expected += 1

    for i, raw in enumerate(lines):
        line, tail = raw.strip(), raw[len(raw.rstrip()):]
        n = numbers[i]
        if n is not None:
            text = _TP_NUMBERED.match(line).group(2) + tail
            after = next((x for x in numbers[i + 1:] if x is not None), None)
            if n == expected and after != n:
                criterion(text)
                continue
            if n == len(elements) + 1:            # a numbered element title
                if crit is not None:
                    crit.extend(pending)
                title, pending = [text], []
                elements.append({"parts": title, "kuk": []})
                continue
        elif line[:1].isupper():                  # "Prepare components 5. Gather all assets"
            m = re.search(rf"\s{expected}\.\s+(\S.*)$", line)
            if m:
                pending.append(line[:m.start()] + " ")
                criterion(m.group(1) + tail)
                continue
        pending.append(raw)
    if crit is not None:
        crit.extend(pending)
    return [{"number": str(e), "title": _tp_join(el["parts"]),
             "kuk": [{"number": f"{e}.{k}", "text": _tp_join(c)} for k, c in enumerate(el["kuk"], 1)]}
            for e, el in enumerate(elements, 1)]


def _tp_drop_explanation(lines: list[str]) -> list[str]:
    """An element table's lines without its explanation sentences: from a line
    that opens one ("Elemen" alone, "Kriteria unjuk kerja menggambarkan") to the
    line that ends it with a full stop, or to the next numbered line."""
    out, skipping = [], False
    for raw in lines:
        s = raw.strip()
        if not skipping and (_TP_EXPLANATION.match(s) or s in ("Elemen", "Elements")):
            skipping = True
        if skipping and (_TP_KUK.match(s) or re.match(r"\d{1,2}\.?\s", s)):
            skipping = False
        if skipping:
            skipping = not s.endswith(".")
            continue
        out.append(raw)
    return out


def _tp_table(lines: list[str]) -> list[dict]:
    """The elements of one language's table, in whichever numbering it uses."""
    lines = _tp_drop_explanation(lines)
    elements = _tp_elements(lines)
    if elements and not any(e["kuk"] for e in elements):
        flat = _tp_flat_elements(lines)
        if any(e["kuk"] for e in flat):
            return flat
    return elements


def _tp_merge(native: list[dict], english: list[dict], problems: list[str]) -> list[dict]:
    """Indonesian elements, completed from the English ones: an element, title or
    KUK that only the English text gives, or that the translation breaks off
    ("... untuk pemikiran dan", the rest never printed), is taken from the
    English text and named in problems."""
    if not native:
        if english:
            problems.append("no Indonesian elements; the English ones are kept")
        return english

    def complete(mine: str, theirs: str) -> bool:
        return bool(theirs) and _tp_unfinished(mine) and not _tp_unfinished(theirs)

    by_number = {e["number"]: e for e in native}
    for e in english:
        mine = by_number.get(e["number"])
        if mine is None:
            native.append(e)
            problems.append(f"element {e['number']} only in English")
            continue
        if not mine["title"] and e["title"]:
            mine["title"] = e["title"]
            problems.append(f"element {e['number']}: title only in English")
        elif complete(mine["title"], e["title"]):
            mine["title"] = e["title"]
            problems.append(f"element {e['number']}: the Indonesian title breaks off; the English one is kept")
        texts = {k["number"]: k for k in mine["kuk"]}
        for k in e["kuk"]:
            if k["number"] not in texts:
                mine["kuk"].append(k)
                problems.append(f"KUK {k['number']} only in English")
            elif not texts[k["number"]]["text"] and k["text"]:
                texts[k["number"]]["text"] = k["text"]
                problems.append(f"KUK {k['number']}: text only in English")
            elif complete(texts[k["number"]]["text"], k["text"]):
                texts[k["number"]]["text"] = k["text"]
                problems.append(f"KUK {k['number']}: the Indonesian text breaks off; the English one is kept")
        mine["kuk"].sort(key=lambda k: int(k["number"].split(".")[1]))
    native.sort(key=lambda e: int(e["number"]))
    return native


def _tp_items(lines: list[str], split: str | None = None) -> list[str]:
    """The bulleted items of a list. Text before the first bullet is an
    introduction ("Evidence of the ability to:"), not an item; a closing
    "Note:" / "Catatan:" ends the list. The full stop or semicolon that closes an
    item of the list is dropped. A bullet printed before the last word of an item
    ("... di browser yang sering" / "• digunakan") does not split it: an item that
    starts in lower case after one that breaks off goes on with it.
    `split` reads lines without a bullet as new items, for a list whose bullets
    the document lost (_tp_aligned decides): "page", a line that starts a page;
    "line", every line after an item that ends with ":" (sub-items)."""
    items: list[list[str]] = []
    loose = after_break = False                   # an item opened without a bullet; a page break seen
    for raw in lines:
        line = raw.strip()
        if _tp_is_feed(raw):
            after_break = True
            continue
        if _TP_NOTE.match(line):
            break
        m = _TP_BULLET.match(line) if line else None
        if m:
            items.append([line[m.end():] + raw[len(raw.rstrip()):]])
            loose = False
        elif line and items and ((split == "page" and after_break) or (
                split == "line" and (loose or _tp_join(items[-1]).endswith(":")))):
            items.append([raw])
            loose = split == "line"
        elif items:
            items[-1].append(raw)
        if line:
            after_break = False
    merged: list[list[str]] = []
    for parts in items:
        text = _tp_join(parts)
        if merged and text[:1].islower() and _tp_unfinished(_tp_join(merged[-1])):
            merged[-1].extend(parts)
        else:
            merged.append(parts)
    return [t for t in (_tp_join(p).rstrip(" .;") for p in merged)
            if t and not _TP_KNOWLEDGE_INTRO.match(t)]


def _tp_aligned(lines: list[str], english: list[str]) -> list[str]:
    """The items of a translated list. The translation is of the English list,
    item by item: when their numbers differ, a reading of lines without a bullet
    as items (_tp_items `split`) that gives the English number is taken, since
    the document lost a bullet at the top of a page ("menjelaskan tanggung jawab
    ...") or printed sub-items without bullets."""
    items = _tp_items(lines)
    if english and len(items) != len(english):
        for split in ("page", "line"):
            alternative = _tp_items(lines, split)
            if len(alternative) == len(english):
                return alternative
    return items


def _tp_listing(native: list[str], english: list[str]) -> list[str]:
    """The Indonesian items of a list, aligned with the English ones; the English
    items when the translation has none."""
    items = _tp_items(english)
    return _tp_aligned(native, items) or items


def _tp_foundation(lines: list[str]) -> list[str]:
    """The described skills of a Foundation Skills table (rows "Reading 1.1, 2.4
    (bullet) Identifies ..."): a bullet, at the start of a line or inside it, opens an
    item; a line that starts with a capital or a digit opens the next row's label
    (skill name, criteria numbers), which is skipped up to its bullet; a line in
    lower case continues the item."""
    items: list[list[str]] = []
    in_label = True
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _TP_INLINE_BULLET.search(line)
        if m:
            items.append([line[m.end():].strip() + raw[len(raw.rstrip()):]])
            in_label = False
        elif not in_label and line[:1].islower():
            items[-1].append(raw)
        else:
            in_label = True
    return [t for t in (_tp_join(p).rstrip(" .;") for p in items) if t]


def _tp_required(lines: list[str]) -> tuple[list[str], list[str]]:
    """(skill lines, knowledge lines) of the older layout's Required Skills and
    Knowledge. Every sub-heading line switches the list, also one the document
    repeats on a continuation page or prints with a bullet ("• Keterampilan yang
    dibutuhkan" under "Ketrampilan yang dibutuhkan")."""
    skills: list[str] = []
    knowledge: list[str] = []
    current: list[str] | None = None
    for raw in lines:
        key = _TP_BULLET.sub("", re.sub(r"\s+", " ", raw).strip()).rstrip(":").strip()
        if _TP_REQ_SKILLS.match(key):
            current = skills
        elif _TP_REQ_KNOWLEDGE.match(key):
            current = knowledge
        elif current is not None:
            current.append(raw)
    return skills, knowledge


def _tp_critical_lines(lines: list[str]) -> list[str]:
    """The lines under the evidence guide's "Critical aspects ..." label, up to the
    next label ("Context of and specific resources", "Method of assessment"),
    which may carry a bullet."""
    start = next((i for i, l in enumerate(lines) if _TP_CRITICAL.match(l.strip())), None)
    if start is None:
        return []
    end = next((j for j in range(start + 1, len(lines))
                if _TP_GUIDE_LABEL.match(_TP_BULLET.sub("", lines[j].strip()))), len(lines))
    return lines[start + 1:end]


def _tp_critical(native: list[str], english: list[str]) -> list[str]:
    """The critical aspects of the older layout's evidence guide: the Indonesian
    ones (aligned with the English), else the English ones."""
    english_items = _tp_items(_tp_critical_lines(english))

    def read(block: list[str], reference: list[str]) -> list[str]:
        items = _tp_aligned(block, reference)
        if not items:  # a single aspect printed without its bullet: the text after "... untuk:"
            intro = max((i for i, l in enumerate(block) if l.strip().endswith(":")), default=None)
            text = _tp_join(block[intro + 1:]).rstrip(" .;") if intro is not None else ""
            items = [text] if text else []
        return items
    return read(_tp_critical_lines(native), english_items) or read(_tp_critical_lines(english), [])


def _tp_unique(items: list[str]) -> list[str]:
    """Items in order, each once (one document prints a translated page twice)."""
    return list(dict.fromkeys(items))


def _tp_range(lines: list[str]) -> dict:
    """The range statement: variables ("Application software may include:") with
    their bulleted items. A variable is the run of lines that ends with ":"; it
    reaches back over lines that do not close a sentence and starts at the first
    capitalised one not glued to the line above; lines before it continue the
    previous item. An item still open (no closing full stop) that goes on in
    lower case to a ":" ("actions arising from ... which may relate to:") is one
    item introducing sub-items, not a variable; nor does a variable start inside
    a parenthesis or after a line that breaks off ("... (such as Environmental
    Protection or" / "Biodiversity Conservation Act):")."""
    split: list[str] = []                         # "Software may include: (bullet) 3D" -> two lines
    for raw in lines:
        m = _TP_INLINE_BULLET.search(raw)
        if m and raw[:m.start()].strip():
            split.extend([raw[:m.start()] + " ", raw[m.start():]])
        else:
            split.append(raw)
    entries: list[tuple[str, list[list[str]]]] = []
    run: list[str] = []

    def glued(line: str) -> bool:
        return bool(line.strip()) and not line[-1:].isspace()

    def opens_variable(i: int, last: list[str] | None) -> bool:
        """Can a variable start at run[i]?"""
        if not run[i].strip()[:1].isupper() or (i > 0 and (glued(run[i - 1]) or _tp_unfinished(run[i - 1]))):
            return False
        before = _tp_join((last or []) + run[:i])
        return before.count("(") <= before.count(")")

    def settle() -> None:
        last = entries[-1][1][-1] if entries and entries[-1][1] else None
        ends = [i for i, r in enumerate(run) if r.strip().endswith(":")]
        if ends:
            k = j = ends[-1]
            while j > 0 and run[j - 1].strip() and not run[j - 1].strip().endswith((".", ";", ":")):
                j -= 1
            start = next((i for i in range(j, k + 1) if opens_variable(i, last)), j)
            if last is not None:
                last.extend(run[:start])
            item = _tp_join(last) if last is not None else ""
            still_open = last is not None and not item.endswith((".", ";"))
            goes_on = (run[start].strip()[:1].islower() or _tp_unfinished(item)
                       or item.count("(") > item.count(")"))
            if not (still_open and goes_on):
                entries.append((_tp_join(run[start:k + 1]), []))
                run.clear()
                return
            del run[:start]
        if last is not None:                      # continuation of the last item
            last.extend(run)
        run.clear()

    for raw in split:
        line = raw.strip()
        m = _TP_BULLET.match(line) if line else None
        if m:
            settle()
            if entries:
                entries[-1][1].append([line[m.end():] + raw[len(raw.rstrip()):]])
        else:
            run.append(raw)
    settle()
    out = {"context": [], "tools": [], "equipment": [], "norms": [], "standards": []}
    for variable, parts in entries:
        items = [t.rstrip(" .;,") for t in (_tp_join(p) for p in parts) if t.rstrip(" .;,")]
        where = next((name for name, rx in _TP_RANGE_LISTS if rx.search(variable)), "context")
        if where in ("tools", "equipment") and _TP_NOT_ARTEFACT.search(variable):
            where = "context"
        if where == "context" or not items:
            out["context"].append(_clean(f"{variable} {'; '.join(items)}"))
        else:
            out[where].extend(items)
    return {name: _tp_unique(items) for name, items in out.items()}


def _tp_description(descriptor: list[str], application: list[str]) -> str:
    """The unit descriptor and the application text, without their row labels
    (one unit prints its label twice: "Deskripsi Unit Deskripsi Unit ini ...").
    A bulleted list inside the text is written as one sentence, its items
    separated by ";" ("ketika bekerja: instalasi ...; sistem operasi ...")."""
    parts = []
    for p in (descriptor, application):
        text = _tp_join(p)
        for _ in range(3):
            text = _TP_LABEL.sub("", text)
        text = re.sub(rf"\s*[{_TP_GLYPHS}]+\s*", "; ", text)
        text = re.sub(r"([:;,.])\s*;\s", r"\1 ", text).strip("; ")
        parts.append(text)
    return " ".join(p for p in parts if p)


def _tp_cover_title(lines: list[str], cover: int | None, heading: int, title_en: str) -> str:
    """The Indonesian title from the cover page (codes and English title removed),
    for a unit whose Indonesian heading was not found."""
    if cover is None or not title_en:
        return ""
    block = []
    for raw in lines[cover:heading]:
        if _TP_PAGE.match(raw) or _tp_is_feed(raw):
            break
        block.append(re.sub(rf"\b{_TP_CODE}\b", " ", raw))
    # one cover carries the words of the next page's "Modification History"
    words = re.sub(r"(?i)\bmodification\s+history\b", " ", _tp_join(block)).split()
    en = title_en.split()
    if [w.lower() for w in words[:len(en)]] == [w.lower() for w in en]:
        return " ".join(words[len(en):])
    return ""


def training_package_toc(text: str) -> list[str]:
    """Unit codes in the document's list of units ("Daftar Unit Kompetensi" up to
    "Uraian Unit Kompetensi"), or [] when there is no such list."""
    i = text.find("Daftar Unit Kompetensi")
    j = text.find("Uraian Unit Kompetensi", i + 1) if i >= 0 else -1
    return re.findall(rf"\b{_TP_CODE}\b", text[i:j]) if j > i else []


# a row of the list of units: "12. BSBSUS201A", "79 ICTWEB416 ...", or the code alone
# (the row number on a line of its own: "51.", "53")
_TP_LIST_ROW = re.compile(rf"^\s*(?:\d{{1,3}}\.?\s+)?({_TP_CODE})\b[ \t]*(.*)$")


def training_package_titles(text: str) -> dict[str, list[str]]:
    """code -> the title lines of the unit in the decree's list of units, as the
    table prints them: the English title, then the Indonesian one. Page numbers,
    the table header repeated on every page and lone row numbers are left out;
    the line of "B. Uraian Unit Kompetensi" ends the list."""
    i = text.find("Daftar Unit Kompetensi")
    j = text.find("Uraian Unit Kompetensi", i + 1) if i >= 0 else -1
    j = text.rfind("\n", i, j) if j > i else -1
    entries: dict[str, list[str]] = {}
    current: list[str] | None = None
    for raw in (text[i:j].split("\n")[1:] if j > i else []):
        s = raw.strip()
        if not s or _TP_PAGE.match(raw) or re.fullmatch(r"\d{1,3}\.?", s) \
                or s.upper().startswith("NO. KODE UNIT"):
            continue
        m = _TP_LIST_ROW.match(raw)
        if m:
            current = entries.setdefault(m.group(1), [])
            if m.group(2).strip():
                current.append(m.group(2))
        elif current is not None:
            current.append(raw)
    return entries


def _tp_list_title(lines: list[str], title_en: str) -> str:
    """The Indonesian title of a unit's row in the list of units. The row's lines
    are its English title, then its Indonesian one; they are cut where the first
    part reads most like the unit's English heading ("" when no cut does)."""
    def words(s: str) -> list[str]:
        return re.findall(r"\w+", s.lower())
    target, best = words(title_en), (0.0, "")
    for k in range(1, len(lines)):
        english, native = _tp_join(lines[:k]), _tp_join(lines[k:])
        score = difflib.SequenceMatcher(None, words(english), target).ratio()
        if native and score > best[0]:
            best = (score, native)
    return best[1] if best[0] >= 0.6 else ""


def is_training_package(text: str) -> bool:
    """A document of training-package units: no SKKNI "KODE UNIT : J." unit, and at
    least one English unit heading ("CODE title" then "Modification History")."""
    return not _UNIT.search(text) and bool(_tp_anchors(text.split("\n")))


def parse_training_package(text: str, source: str = "") -> list[dict]:
    """Units of a standard that adopts training-package units, one dict per unit
    in the schema of parse_units plus `title_en`. The Indonesian text is used;
    a title, element or KUK given only in English is taken from the English text
    and named in `problems`, and so is a title or KUK that still breaks off. The
    Indonesian title is the one of the decree's list of units when the unit is
    in it."""
    lines = text.split("\n")
    anchors = _tp_anchors(lines)
    rows = training_package_titles(text)
    units = []
    for n, (start, heading_code) in enumerate(anchors):
        base = _tp_base(heading_code)
        cover = _tp_cover(lines, anchors[n - 1][0] + 1 if n else 0, start, base)
        end = len(lines)
        if n + 1 < len(anchors):                  # up to the next unit's cover page
            nxt, nxt_code = anchors[n + 1]
            nxt_cover = _tp_cover(lines, start + 1, nxt, _tp_base(nxt_code))
            end = nxt if nxt_cover is None else nxt_cover
        code = _TP_CODE_LINE.match(lines[cover]).group(1) if cover is not None else heading_code
        sec = _tp_sections(lines[start:end], base)

        def part(lang: str, name: str) -> list[str]:
            return sec.get((lang, name), [])

        problems: list[str] = []
        title_en = _tp_join(part("en", "title"))
        row = rows.get(code) or next((v for c, v in rows.items() if _tp_base(c) == base), [])
        title = (_tp_list_title(row, title_en) or _tp_join(part("id", "title"))
                 or _tp_cover_title(lines, cover, start, title_en))
        if not title:
            title = title_en
            problems.append("no Indonesian title; the English one is kept")
        native = _tp_table(part("id", "elements"))
        for e in native:                          # text the translation left in English
            if _tp_language([e["title"]]) == "en":
                problems.append(f"element {e['number']}: title in English in the document")
            problems.extend(f"KUK {k['number']} in English in the document"
                            for k in e["kuk"] if _tp_language([k["text"]]) == "en")
        elements = _tp_merge(native, _tp_table(part("en", "elements")), problems)
        for e in elements:                        # rows the document left empty or cut short
            if not e["title"]:
                problems.append(f"element {e['number']} has no title")
            elif _tp_unfinished(e["title"]):
                problems.append(f"element {e['number']}: the title breaks off")
            problems.extend(f"KUK {k['number']} has no text" for k in e["kuk"] if not k["text"])
            problems.extend(f"KUK {k['number']}: the text breaks off"
                            for k in e["kuk"] if k["text"] and _tp_unfinished(k["text"]))
            e["kuk"] = [k for k in e["kuk"] if k["text"]]
        req_id, req_en = _tp_required(part("id", "required")), _tp_required(part("en", "required"))
        rng = _tp_range(part("id", "range"))
        if not any(rng.values()):
            rng = _tp_range(part("en", "range"))
        units.append({
            "code": code, "title": title, "title_en": title_en,
            "description": (_tp_description(part("id", "descriptor"), part("id", "application"))
                            or _tp_description(part("en", "descriptor"), part("en", "application"))),
            "elements": elements,
            "range": rng,
            "knowledge": _tp_unique(_tp_listing(req_id[1], req_en[1])
                                    or _tp_listing(part("id", "knowledge"), part("en", "knowledge"))),
            "skills": _tp_unique(_tp_listing(req_id[0], req_en[0])
                                 or _tp_listing(part("id", "performance"), part("en", "performance"))
                                 or _tp_foundation(part("id", "foundation"))
                                 or _tp_foundation(part("en", "foundation"))),
            "attitudes": [],
            "critical_aspects": _tp_unique(_tp_critical(part("id", "evidence"), part("en", "evidence"))),
            "prerequisites": sorted({c for c in re.findall(
                rf"\b{_TP_CODE}\b", " ".join(part("id", "prerequisites") + part("en", "prerequisites")))
                if _tp_base(c) != base}),
            "source": source,
            "problems": problems + ([] if elements else ["no element parsed"])
                        + [f"element {e['number']} has no KUK" for e in elements if not e["kuk"]],
        })
    return units


def read_pdf(path: Path) -> str:
    """The text of a PDF. Pages are separated by a line holding only a form feed:
    parse_units reads it as an empty line, parse_training_package as a page break
    (26 pages of the adopted standard carry no page number)."""
    from pypdf import PdfReader
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    return "\n\f\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def parse_text(text: str, source: str = "") -> list[dict]:
    """SKKNI units of a document's text, in whichever layout the text has."""
    if is_training_package(text):
        return parse_training_package(text, source=source)
    return parse_units(text, source=source)


def parse_pdf(path: Path) -> list[dict]:
    path = Path(path)
    return parse_text(read_pdf(path), source=path.name)


ALL_UNITS = SKKNI_DIR / "json" / "skkni_units_all.json"


def decree(text: str) -> tuple[str, str]:
    """(number, year) of the ministerial decree that set the standard. The first
    "KEPUTUSAN MENTERI ... NOMOR n TAHUN y" of the document; a law the decree
    cites ("UU Nomor 13 Tahun 2003") does not match, it is not a KEPUTUSAN."""
    m = _DECREE.search(text)
    return (m.group(1), m.group(2)) if m else ("", "")


def _decree_line(text: str) -> tuple[str, str]:
    """Fallback for decree(): the first "NOMOR n TAHUN y" of the document's
    opening pages (the decree line of its LAMPIRAN)."""
    m = re.search(r"NOMOR\s+(\d+)\s+TAHUN\s+(\d{4})", text[:6000], re.IGNORECASE)
    return (m.group(1), m.group(2)) if m else ("", "")


def parse_all(folder: Path = SKKNI_DIR) -> dict:
    """Every SKKNI PDF of the folder: its decree and its units. A document of
    training-package units (no "KODE UNIT : J." unit) goes to
    parse_training_package; its unit titles are counted from its list of units."""
    docs, units = [], []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        text = read_pdf(pdf)
        number, year = decree(text)
        if is_training_package(text):
            parsed = parse_training_package(text, source=pdf.name)
            n_titles = len(training_package_toc(text)) or len(parsed)
            if not number:
                number, year = _decree_line(text)
        else:
            parsed = parse_units(text, source=pdf.name)
            n_titles = len(re.findall(r"JUDUL UNIT\s*:", text))
        for u in parsed:
            u["decree_number"], u["decree_year"] = number, year
        docs.append({"document": pdf.name, "decree_number": number, "decree_year": year,
                     "n_units": len(parsed), "n_unit_titles_in_text": n_titles,
                     "n_elements": sum(len(u["elements"]) for u in parsed),
                     "n_kuk": sum(len(e["kuk"]) for u in parsed for e in u["elements"]),
                     "n_units_with_a_problem": sum(1 for u in parsed if u["problems"])})
        units.extend(parsed)
    return {"parser_version": PARSER_VERSION, "documents": docs, "units": units}


def load_units(path: Path | None = None) -> list[dict]:
    if path is None:
        path = ALL_UNITS if ALL_UNITS.is_file() else SKKNI_DIR / "json" / "skkni_units_pemrograman.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))["units"]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("pdf")
    p.add_argument("--out", default=None)
    sub.add_parser("parse-all", help="every PDF in DATA/master_skkni -> skkni_units_all.json")
    args = ap.parse_args(argv)
    if args.cmd == "parse-all":
        out = parse_all()
        ALL_UNITS.parent.mkdir(parents=True, exist_ok=True)
        ALL_UNITS.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        for d in out["documents"]:
            flag = "" if d["n_units"] == d["n_unit_titles_in_text"] else "  <- units != titles"
            print(f"{d['document'][:58]:58s} {d['decree_number']:>4s}/{d['decree_year']:4s} "
                  f"units {d['n_units']:3d} (titles {d['n_unit_titles_in_text']:3d}) elements "
                  f"{d['n_elements']:4d} KUK {d['n_kuk']:4d} problems {d['n_units_with_a_problem']}{flag}")
        print(f"{len(out['units'])} units -> {ALL_UNITS}")
        return 0
    units = parse_pdf(Path(args.pdf))
    out = Path(args.out) if args.out else SKKNI_DIR / "json" / "skkni_units_pemrograman.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"parser_version": PARSER_VERSION, "source": Path(args.pdf).name,
                               "n_units": len(units), "units": units}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    bad = [u for u in units if u["problems"]]
    n_kuk = sum(len(e["kuk"]) for u in units for e in u["elements"])
    print(f"{len(units)} units, {sum(len(u['elements']) for u in units)} elements, {n_kuk} KUK; "
          f"{len(bad)} with a problem -> {out}")
    for u in bad:
        print("  ", u["code"], u["title"], u["problems"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
