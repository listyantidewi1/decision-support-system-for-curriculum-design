"""Tests for competency.skkni: the classic SKKNI layout and the layout of a
standard that adopts training-package units (English and Indonesian versions
of every unit, pages alternating between the two). Every code, title and
sentence here is invented; no PDF is read.

Run:  .venv-ml\\Scripts\\python.exe tests\\test_skkni.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competency import skkni as SK                                            # noqa: E402

SKKNI_TEXT = """KODE UNIT :   J.620100.099.01
JUDUL UNIT : Menguji Contoh Unit
DESKRIPSI UNIT : Unit ini menentukan kompetensi, pengetahuan dan
sikap kerja yang diperlukan untuk menguji contoh.
ELEMEN KOMPETENSI KRITERIA UNJUK KERJA
1. Menyiapkan
contoh uji
1.1 Kebutuhan contoh diidentifikasi
sesuai spesifikasi.
1.2 Data contoh disiapkan.
12
2. Melaksanakan uji
2.1 Uji dijalankan sesuai rencana.
3. Melaporkan hasil
BATASAN VARIABEL
1. Konteks variabel
1.1 Contoh adalah program kecil.
2. Peralatan dan perlengkapan
2.1 Peralatan
2.1.1 Perangkat lunak pengujian
2.2 Perlengkapan
(Tidak ada.)
4. Norma dan standar
4.1 Norma
4.1.1 Etika profesi
4.2 Standar
4.2.1 Standar pengujian
PANDUAN PENILAIAN
2. Persyaratan kompetensi
2.1 J.620100.004.01 : Menggunakan Struktur Data
3. Pengetahuan dan keterampilan yang dibutuhkan
3.1 Pengetahuan
3.1.1 Dasar pengujian perangkat lunak
3.2 Keterampilan
3.2.1 Menjalankan alat uji
4. Sikap kerja yang diperlukan
4.1 Teliti
4.2 Cermat
5. Aspek kritis
5.1 Ketepatan menjalankan uji
"""


def _pdf(lines: list[str]) -> str:
    """Lines as pypdf gives them: a space at the end of every line, none where the
    PDF cut a word (a line written with a final "|"). "@" and "%" stand for two
    of the document's bullet glyphs (U+F0B7 and the black circle U+25CF). "<FF>"
    is the form-feed line read_pdf writes between two pages."""
    out = []
    for line in lines:
        if line == "<FF>":
            out.append("\f")
            continue
        line = line.replace("@", chr(0xF0B7)).replace("%", chr(0x25CF))
        out.append(line[:-1] if line.endswith("|") else line + " ")
    return "\n".join(out)


def _swap(lines: list[str], old: list[str], new: list[str]) -> list[str]:
    """`lines` with the run of lines `old` (which must occur) replaced by `new`."""
    for i in range(len(lines) - len(old) + 1):
        if lines[i:i + len(old)] == old:
            return lines[:i] + new + lines[i + len(old):]
    raise ValueError(f"not found: {old}")


# A unit in the newer layout. The element tables go on over a page each, without
# a heading; an Indonesian element title and a knowledge item are cut by the PDF;
# one criterion uses the translation's decimal comma ("1,2").
NEWER = [
    "ICTXMP401 Build sample",                     # the cover page
    "widgets",
    "Membangun widget contoh",
    "", "17", "",
    "ICTXMP401 Build sample widgets",
    "Modification History",
    "Release Comments",
    "Release 1 This version first released with the ICT Training Package.",
    "Application",
    "This unit describes the skills and knowledge required to build sample widgets.",
    "Unit Sector",
    "Programming",
    "Elements and Performance Criteria",
    "ELEMENT PERFORMANCE CRITERIA",
    "Elements describe the essential outcomes.",
    "Performance criteria describe the performance needed to demonstrate achievement of the element.",
    "1 Plan the",
    "widget",
    "1.1 Identify the requirements of the widget with",
    "the client",
    "1.2 Select the tools for the widget",
    "", "18", "",
    "ICTXMP401 Membangun widget contoh",
    "Riwayat Perubahan",
    "Rilis Komentar",
    "Rilis 1 Versi ini pertama kali dirilis dengan Paket Pelatihan TIK.",
    "Penerapan",
    "Unit ini menjelaskan keterampilan dan pengetahuan yang diperlukan untuk",
    "membangun widget contoh.",
    "Sektor Unit",
    "Pemrograman",
    "Elemen dan Kriteria Unjuk Kerja",
    "ELEMEN KRITERIA UNJUK KERJA",
    "Elemen",
    "menggambarkan",
    "hasil penting.",
    "Kriteria unjuk kerja menggambarkan unjuk kerja yang diperlukan.",
    "1. Merencana|",
    "kan widget",
    "1.1 Mengidentifikasi kebutuhan widget bersama",
    "klien",
    "1,2 Memilih alat untuk widget",
    "", "19", "",
    "2 Build the widget",                         # English page, no heading
    "2.1 Write the code of the widget in line with the plan",
    "2.2 Test the widget and record the results of the test",
    "", "20", "",
    "2. Membangun widget",                        # Indonesian page, no heading
    "2.1 Menulis kode widget sesuai dengan rencana",
    "2.2 Menguji widget dan mencatat hasil dari pengujian",
    "", "21", "",
    "Foundation Skills",
    "This section describes language, literacy, numeracy and employment skills.",
    "Skill Performance",
    "Criteria",
    "Description",
    "Reading 1.1, 1.2 @ Interprets the needs of the client",
    "from documents",
    "Writing 2.2 @ Records the results of tests",
    "", "22", "",
    "Keterampilan Dasar",
    "Bagian ini menjelaskan kemampuan bahasa, literasi, numerik dan keterampilan kerja.",
    "Keterampilan Kriteria",
    "Unjuk Kerja",
    "Deskripsi",
    "Membaca 1.1, 1.2 @ Menafsirkan kebutuhan klien",
    "dari dokumen",
    "Menulis 2.2 @ Mencatat hasil pengujian",
    "", "23", "",
    "Assessment Requirements for ICTXMP401 Build sample",
    "widgets",
    "Modification History",
    "Release Comments",
    "Release 1 This version first released with the ICT Training Package.",
    "Performance Evidence",
    "Evidence of the ability to:",
    "@ build two widgets",
    "@ test a widget with the client.",
    "Note: If a specific volume or frequency is not stated, then evidence must be provided at least once.",
    "Knowledge Evidence",
    "To complete the unit requirements safely and effectively, the individual must:",
    "@ explain the parts of a widget.",
    "Assessment Conditions",
    "Skills must be demonstrated in a workplace.",
    "", "24", "",
    "Persyaratan Penilaian untuk ICTXMP401 Membangun",
    "widget contoh",
    "Riwayat Perubahan",
    "Rilis Komentar",
    "Rilis 1 Versi ini pertama kali dirilis dengan Paket Pelatihan TIK.",
    "Bukti Unjuk Kerja",
    "Bukti kemampuan untuk:",
    "% membangun dua widget",
    "% menguji sebuah widget bersama klien.",
    "Catatan: Jika volume atau frekuensi tertentu tidak disebutkan, maka bukti harus diberikan setidaknya satu kali.",
    "Penilaian Pengetahuan",
    "Untuk melengkapi persyaratan unit dengan aman dan efektif, individu harus:",
    "% menjelaskan bagian-|",
    "bagian sebuah widget.",
    "Ketentuan Penilaian",
    "Keterampilan harus ditunjukkan di tempat kerja.",
    "", "25", "",
]

# A unit in the older layout (code with a final letter): descriptor and
# application with their row labels, a prerequisite, required skills and
# knowledge, an evidence guide with critical aspects and a range statement. One
# line of the descriptor reads like a heading in lower case; the Indonesian
# element table repeats its header and explanation on a page after its criteria.
OLDER = [
    "ICAXMP301A Maintain sample",
    "records",
    "Memelihara catatan contoh",
    "", "30", "",
    "ICAXMP301A Maintain sample records",
    "Modification History",
    "Not applicable.",
    "Unit Descriptor",
    "Unit descriptor This unit describes the performance outcomes, skills and",
    "knowledge required to maintain sample records.",
    "Application of the Unit",
    "Application of the",
    "unit",
    "This unit applies to clerks who keep the records of a team.",
    "Pre-Requisites",
    "Prerequisite units",
    "ICAXMP101A Use sample files",
    "", "31", "",
    "ICAXMP301A Memelihara catatan contoh",
    "Riwayat Perubahan",
    "Tidak berlaku.",
    "Deskripsi Unit",
    "Deskripsi unit Unit ini menjelaskan hasil unjuk kerja,",
    "keterampilan dan pengetahuan yang diperlukan",
    "untuk memelihara catatan contoh.",
    "Penerapan Unit",
    "Aplikasi unit Unit ini berlaku untuk petugas yang menyimpan catatan sebuah tim.",
    "Prasyarat",
    "Unit prasyarat",
    "ICAXMP101A Menggunakan berkas contoh",
    "", "32", "",
    "Elements and Performance Criteria",
    "ELEMENT PERFORMANCE CRITERIA",
    "1 Check the",
    "records",
    "1.1 Check each record against the register",
    "1.2 Report missing records to the supervisor",
    "2 Archive the",
    "records",
    "2.1 Archive closed records in line with the policy of the team",
    "", "33", "",
    "Elemen dan Kriteria Unjuk Kerja",
    "ELEMEN KRITERIA UNJUK KERJA",
    "1. Memeriksa",
    "catatan",
    "1.1 Memeriksa setiap catatan terhadap register",
    "1.2 Melaporkan catatan yang hilang kepada",
    "atasan",
    "2. Mengarsip|",
    "kan catatan",
    "2.1 Mengarsipkan catatan yang ditutup sesuai",
    "dengan kebijakan tim",
    "", "34", "",
    "ELEMEN KRITERIA UNJUK KERJA",                # the table's explanation, printed again
    "Elemen menentukan hasil yang",
    "penting.",
    "Kriteria unjuk kerja menentukan tingkat kinerja",
    "yang diperlukan untuk menunjukkan pencapaian elemen.",
    "", "35", "",
    "Required Skills and Knowledge",
    "This section describes the skills and knowledge required for this unit.",
    "Required skills",
    "@ literacy skills to read the register",
    "@ communication skills to report to the supervisor",
    "Required knowledge",
    "@ record-keeping policies of the organisation.",
    "", "35", "",
    "Keterampilan dan Pengetahuan yang Dibutuhkan",
    "Bagian ini menjelaskan keterampilan dan pengetahuan yang dibutuhkan untuk unit ini.",
    "Keterampilan yang dibutuhkan",
    "@ keterampilan literasi untuk membaca register",
    "@ keterampilan komunikasi untuk melapor kepada",
    "atasan",
    "Pengetahuan yang dibutuhkan",
    "@ kebijakan penyimpanan catatan dari organisasi.",
    "", "36", "",
    "Evidence Guide",
    "The evidence guide provides advice on assessment and must be read with the criteria.",
    "Critical aspects for",
    "assessment and",
    "evidence required",
    "Evidence of the ability to:",
    "@ check records against a register",
    "@ archive closed records.",
    "Context of and",
    "specific resources",
    "Assessment must ensure access to:",
    "@ a register",
    "", "37", "",
    "Panduan Penilaian",
    "Panduan penilaian memberikan nasihat tentang penilaian dan harus dibaca dengan kriteria.",
    "Aspek-aspek",
    "kritis untuk",
    "penilaian",
    "Bukti kemampuan untuk:",
    "@ memeriksa catatan terhadap register",
    "@ mengarsipkan catatan yang ditutup.",
    "Konteks dan",
    "sumber daya",
    "Penilaian harus memastikan akses ke:",
    "@ sebuah register",
    "", "38", "",
    "Range Statement",
    "The range statement relates to the unit of competency as a whole.",
    "Software may include: @ spreadsheets",
    "@ record databases.",
    "Registers may",
    "include:",
    "@ paper registers",
    "@ electronic registers.",
    "", "39", "",
    "Batasan Variabel",
    "Batasan variabel berkaitan dengan unit kompetensi secara keseluruhan.",
    "Perangkat lunak dapat mencakup: @ lembar kerja",
    "@ basis data catatan.",
    "Register dapat",
    "mencakup:",
    "@ register kertas",
    "@ register elektronik.",
    "Sektor Unit",
    "Administrasi",
    "", "40", "",
]

# The older numbering of two screen-media units: element titles without a number
# (English) or with their own number (Indonesian), criteria numbered through the unit.
THROUGH_NUMBERED = [
    "CUFXMP402A Make sample effects",
    "Modification History",
    "Not applicable.",
    "Elements and Performance Criteria",
    "ELEMENT PERFORMANCE CRITERIA",
    "Plan the",
    "work",
    "1. Read the brief of the client",
    "2. List the tasks in order",
    "3. Agree the plan with the producer",
    "Do the work 4. Carry out the tasks in the plan",
    "5. Check the",
    "result",
    "", "50", "",
    "CUFXMP402A Membuat efek contoh",
    "Riwayat Perubahan",
    "Tidak berlaku.",
    "Elemen dan Kriteria Unjuk Kerja",
    "1. Merencanakan",
    "pekerjaan",
    "1. Membaca ringkasan dari klien",
    "2. Mendaftar tugas secara berurutan",
    "3. Menyepakati rencana dengan produser",
    "2. Melakukan",
    "pekerjaan",
    "4. Melaksanakan tugas sesuai rencana",
    "5. Memeriksa hasil",
    "", "51", "",
]

LIST_OF_UNITS = [
    "BAB II",
    "STANDAR KOMPETENSI KERJA",
    "A. Daftar Unit Kompetensi",
    "NO. KODE UNIT JUDUL UNIT KOMPETENSI",
    "1. ICAXMP301A",
    "Maintain Sample Records",
    "Memelihara Catatan Contoh",
    "2. ICTXMP401",
    "Build Sample Widgets",
    "Membangun Widget Contoh",
    "B. Uraian Unit Kompetensi",
]


def test_a_classic_skkni_unit_is_still_read_by_parse_units():
    (u,) = SK.parse_units(SKKNI_TEXT, source="invented")
    assert u["code"] == "J.620100.099.01" and u["title"] == "Menguji Contoh Unit"
    assert [e["title"] for e in u["elements"]] == ["Menyiapkan contoh uji", "Melaksanakan uji",
                                                   "Melaporkan hasil"]
    assert [k["text"] for k in u["elements"][0]["kuk"]] == [
        "Kebutuhan contoh diidentifikasi sesuai spesifikasi.", "Data contoh disiapkan."]
    assert u["range"]["tools"] == ["Perangkat lunak pengujian"]
    assert u["knowledge"] == ["Dasar pengujian perangkat lunak"] and u["attitudes"] == ["Teliti", "Cermat"]
    assert u["problems"] == ["element 3 has no KUK"]
    assert "title_en" not in u                    # the classic schema is unchanged


def test_a_newer_layout_unit_is_read_in_indonesian_with_its_english_title():
    (u,) = SK.parse_training_package(_pdf(NEWER), source="invented.pdf")
    assert list(u) == ["code", "title", "title_en", "description", "elements", "range", "knowledge",
                       "skills", "attitudes", "critical_aspects", "prerequisites", "source", "problems"]
    assert (u["code"], u["title"], u["title_en"]) == ("ICTXMP401", "Membangun widget contoh",
                                                     "Build sample widgets")
    assert u["description"] == ("Unit ini menjelaskan keterampilan dan pengetahuan yang diperlukan "
                                "untuk membangun widget contoh.")
    # a word cut by the PDF is glued back; "1,2" is criterion 1.2; the second element
    # comes from a page without a heading, in the right language
    assert u["elements"] == [
        {"number": "1", "title": "Merencanakan widget",
         "kuk": [{"number": "1.1", "text": "Mengidentifikasi kebutuhan widget bersama klien"},
                 {"number": "1.2", "text": "Memilih alat untuk widget"}]},
        {"number": "2", "title": "Membangun widget",
         "kuk": [{"number": "2.1", "text": "Menulis kode widget sesuai dengan rencana"},
                 {"number": "2.2", "text": "Menguji widget dan mencatat hasil dari pengujian"}]}]
    assert u["skills"] == ["membangun dua widget", "menguji sebuah widget bersama klien"]
    assert u["knowledge"] == ["menjelaskan bagian-bagian sebuah widget"]     # intro and "Catatan:" left out
    assert u["attitudes"] == [] and u["critical_aspects"] == [] and u["prerequisites"] == []
    assert not any(u["range"].values())
    assert u["source"] == "invented.pdf" and u["problems"] == []


def test_page_numbers_and_repeated_table_headers_do_not_leak_into_titles_or_criteria():
    (u,) = SK.parse_training_package(_pdf(NEWER))
    text = json.dumps([u["title"], u["title_en"], u["elements"]], ensure_ascii=False)
    for junk in ("ELEMEN", "KRITERIA", "describe", "menggambarkan", "hasil penting", "Modification",
                 "Riwayat", "17", "18", "19", "20", "21"):
        assert junk not in text, junk


def test_an_older_layout_unit_gives_required_skills_knowledge_critical_aspects_and_range():
    (u,) = SK.parse_training_package(_pdf(OLDER))
    assert (u["code"], u["title"], u["title_en"]) == ("ICAXMP301A", "Memelihara catatan contoh",
                                                     "Maintain sample records")
    # row labels dropped; the lower-case "keterampilan dan pengetahuan yang diperlukan"
    # inside the descriptor is text, not the heading of Required Skills and Knowledge
    assert u["description"] == ("Unit ini menjelaskan hasil unjuk kerja, keterampilan dan pengetahuan "
                                "yang diperlukan untuk memelihara catatan contoh. Unit ini berlaku untuk "
                                "petugas yang menyimpan catatan sebuah tim.")
    assert [(e["number"], e["title"]) for e in u["elements"]] == [("1", "Memeriksa catatan"),
                                                                 ("2", "Mengarsipkan catatan")]
    assert [k["text"] for k in u["elements"][0]["kuk"]] == [
        "Memeriksa setiap catatan terhadap register", "Melaporkan catatan yang hilang kepada atasan"]
    assert u["elements"][1]["kuk"] == [{"number": "2.1", "text": "Mengarsipkan catatan yang ditutup "
                                                                 "sesuai dengan kebijakan tim"}]
    assert u["skills"] == ["keterampilan literasi untuk membaca register",
                           "keterampilan komunikasi untuk melapor kepada atasan"]
    assert u["knowledge"] == ["kebijakan penyimpanan catatan dari organisasi"]
    assert u["critical_aspects"] == ["memeriksa catatan terhadap register",
                                     "mengarsipkan catatan yang ditutup"]
    assert u["range"] == {"context": ["Register dapat mencakup: register kertas; register elektronik"],
                          "tools": ["lembar kerja", "basis data catatan"], "equipment": [],
                          "norms": [], "standards": []}
    assert u["prerequisites"] == ["ICAXMP101A"] and u["attitudes"] == [] and u["problems"] == []


def test_a_table_explanation_printed_after_the_criteria_does_not_join_the_last_criterion():
    (u,) = SK.parse_training_package(_pdf(OLDER))
    assert u["elements"][-1]["kuk"][-1]["text"] == "Mengarsipkan catatan yang ditutup sesuai dengan kebijakan tim"
    assert "menentukan" not in json.dumps(u["elements"])


def test_a_knowledge_list_the_translation_put_under_another_heading_is_still_knowledge():
    # the translation heads its knowledge evidence "Ketentuan Penilaian" (assessment
    # conditions) and bullets the opening sentence; that sentence tells the section
    lines = ["Ketentuan Penilaian" if l == "Penilaian Pengetahuan" else
             "% " + l if l.startswith("Untuk melengkapi persyaratan unit") else l for l in NEWER]
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["knowledge"] == ["menjelaskan bagian-bagian sebuah widget"]


def test_a_criterion_missing_in_the_translation_is_taken_from_english_and_reported():
    lines = [l for l in NEWER if not l.startswith("2.2 Menguji")]
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["elements"][1]["kuk"][1] == {"number": "2.2",
                                          "text": "Test the widget and record the results of the test"}
    assert u["problems"] == ["KUK 2.2 only in English"]


def test_criteria_numbered_through_the_unit_are_numbered_per_element():
    (u,) = SK.parse_training_package(_pdf(THROUGH_NUMBERED))
    assert u["elements"] == [
        {"number": "1", "title": "Merencanakan pekerjaan",
         "kuk": [{"number": "1.1", "text": "Membaca ringkasan dari klien"},
                 {"number": "1.2", "text": "Mendaftar tugas secara berurutan"},
                 {"number": "1.3", "text": "Menyepakati rencana dengan produser"}]},
        {"number": "2", "title": "Melakukan pekerjaan",
         "kuk": [{"number": "2.1", "text": "Melaksanakan tugas sesuai rencana"},
                 {"number": "2.2", "text": "Memeriksa hasil"}]}]
    assert u["problems"] == []                    # the English table gives the same two elements


def test_without_assessment_requirements_the_skills_come_from_the_foundation_skills():
    cut = next(i for i, l in enumerate(NEWER) if l.startswith("Assessment Requirements"))
    (u,) = SK.parse_training_package(_pdf(NEWER[:cut]))
    assert u["skills"] == ["Menafsirkan kebutuhan klien dari dokumen", "Mencatat hasil pengujian"]
    assert u["knowledge"] == []


def test_the_layout_is_told_by_the_text_and_the_list_of_units_is_read():
    adopted = _pdf(LIST_OF_UNITS + OLDER + NEWER)
    assert SK.is_training_package(adopted) and not SK.is_training_package(SKKNI_TEXT)
    assert SK.training_package_toc(adopted) == ["ICAXMP301A", "ICTXMP401"]
    assert [u["code"] for u in SK.parse_text(adopted)] == ["ICAXMP301A", "ICTXMP401"]
    assert [u["code"] for u in SK.parse_text(SKKNI_TEXT)] == ["J.620100.099.01"]
    # the unit after another one ends where its cover page begins
    older, newer = SK.parse_text(adopted)
    assert older["range"]["context"] == ["Register dapat mencakup: register kertas; register elektronik"]
    assert newer["title"] == "Membangun Widget Contoh"     # the list's title, see the test below
    assert SK.parse_training_package(SKKNI_TEXT) == []


def test_parse_all_reads_both_layouts_and_the_decree_of_each_document():
    texts = {"adopted.pdf": "LAMPIRAN\nPERATURAN MENTERI KOMUNIKASI\nNOMOR 7 TAHUN 2021\n"
                            + _pdf(LIST_OF_UNITS + OLDER + NEWER),
             "classic.pdf": "LAMPIRAN\nKEPUTUSAN MENTERI KETENAGAKERJAAN\nREPUBLIK INDONESIA\n"
                            "NOMOR 282 TAHUN 2016\n" + SKKNI_TEXT}
    original = SK.read_pdf
    with tempfile.TemporaryDirectory() as tmp:
        for name in texts:
            (Path(tmp) / name).write_bytes(b"%PDF-1.4 invented")
        SK.read_pdf = lambda path: texts[Path(path).name]
        try:
            out = SK.parse_all(Path(tmp))
        finally:
            SK.read_pdf = original
    docs = {d["document"]: d for d in out["documents"]}
    adopted, classic = docs["adopted.pdf"], docs["classic.pdf"]
    # no "KEPUTUSAN MENTERI" in the adopted document: its decree is read from its NOMOR line
    assert (adopted["decree_number"], adopted["decree_year"]) == ("7", "2021")
    assert (adopted["n_units"], adopted["n_unit_titles_in_text"]) == (2, 2)
    assert (adopted["n_elements"], adopted["n_kuk"], adopted["n_units_with_a_problem"]) == (4, 7, 0)
    assert (classic["decree_number"], classic["decree_year"], classic["n_units"]) == ("282", "2016", 1)
    assert classic["n_unit_titles_in_text"] == 1
    by_code = {u["code"]: u for u in out["units"]}
    assert by_code["ICTXMP401"]["decree_year"] == "2021" and by_code["ICTXMP401"]["title_en"]
    assert by_code["J.620100.099.01"]["decree_number"] == "282"


# ------------------------------------------------ regressions of the 2026-11-20 check
# One test per kind of defect an independent check of the real document found; the
# unit it was found in is named in each comment. All text is invented.

def test_an_element_title_cell_that_goes_on_after_a_page_break_ends_the_title():
    # ICTTEN810: the title cell broke off at "untuk", and its end, printed at the top
    # of the next page, was glued onto the criterion
    lines = _swap(NEWER, ["2. Membangun widget", "2.1 Menulis kode widget sesuai dengan rencana"],
                  ["2. Membangun widget", "untuk", "2.1 Menulis kode widget sesuai dengan rencana",
                   "", "99", "", "klien", ""])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["elements"][1]["title"] == "Membangun widget untuk klien"
    assert [k["text"] for k in u["elements"][1]["kuk"]] == [
        "Menulis kode widget sesuai dengan rencana", "Menguji widget dan mencatat hasil dari pengujian"]
    assert u["problems"] == []


# the knowledge evidence over two pages in each language; the second Indonesian page
# is the English one left untranslated but for one item, so its words read as English
TWIN_ASSESSMENT = [
    "Assessment Requirements for ICTXMP401 Build sample",
    "widgets",
    "Modification History",
    "Release Comments",
    "Release 1 This version first released with the ICT Training Package.",
    "Performance Evidence",
    "Evidence of the ability to:",
    "@ build two widgets.",
    "Knowledge Evidence",
    "To complete the unit requirements safely and effectively, the individual must:",
    "@ explain the parts of a widget",
    "@ describe the materials of a widget:",
    "", "24", "",
    "Persyaratan Penilaian untuk ICTXMP401 Membangun",
    "widget contoh",
    "Riwayat Perubahan",
    "Rilis Komentar",
    "Rilis 1 Versi ini pertama kali dirilis dengan Paket Pelatihan TIK.",
    "Bukti Unjuk Kerja",
    "Bukti kemampuan untuk:",
    "% membangun dua widget.",
    "Penilaian Pengetahuan",
    "Untuk melengkapi persyaratan unit dengan aman dan efektif, individu harus:",
    "% menjelaskan bagian-bagian sebuah widget",
    "% menguraikan bahan dari sebuah widget:",
    "", "25", "",
    "@ frames of wood and metal",
    "@ fasteners for the frames",
    "@ covers of plastic",
    "@ seals of rubber and silicone.",
    "Assessment Conditions",
    "Skills must be demonstrated in a workplace.",
    "", "26", "",
    "% frames of wood and metal",
    "% fasteners for the frames",
    "% penutup dari plastik",
    "% seals of rubber and silicone.",
    "Ketentuan Penilaian",
    "Keterampilan harus ditunjukkan di tempat kerja.",
    "", "27", "",
]


def test_a_page_that_repeats_the_page_before_it_is_its_twin_in_the_other_language():
    # HLTAID003, CUSSOU302A: such a page went to the English section and was lost
    cut = NEWER.index("Assessment Requirements for ICTXMP401 Build sample")
    (u,) = SK.parse_training_package(_pdf(NEWER[:cut] + TWIN_ASSESSMENT))
    assert u["knowledge"] == ["menjelaskan bagian-bagian sebuah widget", "menguraikan bahan dari sebuah widget:",
                              "frames of wood and metal", "fasteners for the frames", "penutup dari plastik",
                              "seals of rubber and silicone"]
    assert u["skills"] == ["membangun dua widget"]


def test_pages_without_a_number_are_told_apart_by_the_form_feed_between_them():
    # CUVPHI519A: the range pages carry no page number; the English ones were read
    # as the Indonesian section going on
    cut = OLDER.index("Range Statement")
    lines = OLDER[:cut] + [
        "Range Statement",
        "The range statement relates to the unit of competency as a whole.",
        "Software may include: @ spreadsheets",
        "@ record databases.",
        "<FF>",
        "Batasan Variabel",
        "Batasan variabel berkaitan dengan unit kompetensi secara keseluruhan.",
        "Perangkat lunak dapat mencakup: @ lembar kerja",
        "@ basis data catatan.",
        "<FF>",
        "Registers may",
        "include:",
        "@ paper registers kept by the team",
        "@ electronic registers of the team.",
        "<FF>",
        "Register dapat",
        "mencakup:",
        "@ register kertas yang disimpan tim",
        "@ register elektronik dari tim.",
        "", "40", ""]
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["range"]["context"] == [
        "Register dapat mencakup: register kertas yang disimpan tim; register elektronik dari tim"]
    assert u["range"]["tools"] == ["lembar kerja", "basis data catatan"]


def test_a_criterion_the_translation_breaks_off_is_taken_from_english_and_reported():
    # BSBINN801: "Evaluasi konteks keseluruhan untuk pemikiran dan" was kept as it was
    lines = [{"1,2 Memilih alat untuk widget": "1,2 Memilih alat untuk"}.get(l, l) for l in NEWER]
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["elements"][0]["kuk"][1] == {"number": "1.2", "text": "Select the tools for the widget"}
    assert u["problems"] == ["KUK 1.2: the Indonesian text breaks off; the English one is kept"]
    # without a complete English criterion the fragment stays, and is reported
    lines = [{"1,2 Memilih alat untuk widget": "1,2 Memilih alat untuk",
              "1.2 Select the tools for the widget": "1.2 Select the tools for"}.get(l, l) for l in NEWER]
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["elements"][0]["kuk"][1]["text"] == "Memilih alat untuk"
    assert u["problems"] == ["KUK 1.2: the text breaks off"]


def test_repeated_and_bulleted_sub_headings_and_an_empty_sub_bullet_are_not_items():
    # CUFANM402A: the sub-heading repeated on a continuation page was glued onto an
    # item; ICAICT408A: a bulleted copy of the sub-heading was an item, and an empty
    # "o" sub-bullet was glued onto the last skill
    lines = _swap(OLDER, ["@ record-keeping policies of the organisation."],
                  ["@ record-keeping policies of the organisation", "@ archiving procedures of the team."])
    lines = _swap(lines, ["Keterampilan yang dibutuhkan", "@ keterampilan literasi untuk membaca register",
                          "@ keterampilan komunikasi untuk melapor kepada", "atasan",
                          "Pengetahuan yang dibutuhkan", "@ kebijakan penyimpanan catatan dari organisasi."],
                  ["Keterampilan yang dibutuhkan", "@ Keterampilan yang dibutuhkan",
                   "@ keterampilan literasi untuk membaca register",
                   "@ keterampilan komunikasi untuk melapor kepada", "atasan", "o",
                   "Pengetahuan yang dibutuhkan", "@ kebijakan penyimpanan catatan dari organisasi",
                   "", "99", "",
                   "KETERAMPILAN DAN PENGETAHUAN YANG DIBUTUHKAN",
                   "Pengetahuan yang dibutuhkan",
                   "@ prosedur pengarsipan dari tim."])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["skills"] == ["keterampilan literasi untuk membaca register",
                           "keterampilan komunikasi untuk melapor kepada atasan"]
    assert u["knowledge"] == ["kebijakan penyimpanan catatan dari organisasi", "prosedur pengarsipan dari tim"]


def test_an_item_that_lost_its_bullet_at_the_top_of_a_page_is_split_off_by_the_english_count():
    # BSBSMB407: two knowledge items were read as one
    lines = _swap(NEWER, ["@ explain the parts of a widget."],
                  ["@ explain the parts of a widget", "@ describe the tests of a widget."])
    lines = _swap(lines, ["bagian sebuah widget."],
                  ["bagian sebuah widget", "", "99", "", "menguraikan pengujian sebuah widget."])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["knowledge"] == ["menjelaskan bagian-bagian sebuah widget", "menguraikan pengujian sebuah widget"]


def test_a_bullet_printed_before_the_last_word_of_an_item_does_not_split_it():
    # ICTWEB416: "... di browser yang sering" / "• digunakan" were two items; an item
    # after one that ends a phrase ("bahan:" / "kayu") stays an item
    lines = _swap(NEWER, ["@ explain the parts of a widget."],
                  ["@ explain the commonly used parts of a widget", "@ describe the materials:", "@ wood",
                   "@ metal."])
    lines = _swap(lines, ["% menjelaskan bagian-|", "bagian sebuah widget."],
                  ["% menjelaskan bagian-bagian widget yang sering", "% digunakan", "% menguraikan bahan:",
                   "% kayu", "% logam."])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["knowledge"] == ["menjelaskan bagian-bagian widget yang sering digunakan", "menguraikan bahan:",
                              "kayu", "logam"]


def test_sub_items_printed_without_bullets_are_items_when_the_english_list_says_so():
    # BSBSMB405: the sub-items ran into their parent item as one string
    lines = _swap(NEWER, ["@ build two widgets"],
                  ["@ build two widgets, including:", "@ cutting the materials", "@ assembling the parts"])
    lines = _swap(lines, ["% membangun dua widget"],
                  ["% membangun dua widget, termasuk:", "memotong bahan", "merakit bagian-bagian"])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["skills"] == ["membangun dua widget, termasuk:", "memotong bahan", "merakit bagian-bagian",
                           "menguji sebuah widget bersama klien"]


def test_a_bulleted_label_of_the_evidence_guide_ends_the_critical_aspects():
    # ICAICT408A: "• Konteks dan sumber daya khusus untuk penilaian" was a fifth aspect
    lines = _swap(OLDER, ["@ mengarsipkan catatan yang ditutup.", "Konteks dan"],
                  ["@ mengarsipkan catatan yang ditutup.", "@ Konteks dan sumber daya khusus untuk penilaian",
                   "Konteks dan"])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["critical_aspects"] == ["memeriksa catatan terhadap register", "mengarsipkan catatan yang ditutup"]


def test_a_range_variable_about_sites_or_methods_is_context_not_tools():
    # ICAICT103A: "Situs aplikasi internet" (sites) was filed under tools because of
    # "aplikasi", next to the browser software; methods the same (ICAPRG502A)
    lines = _swap(OLDER, ["@ register elektronik.", "Sektor Unit"],
                  ["@ register elektronik.", "Situs aplikasi internet dapat mencakup: @ perbankan", "@ belanja.",
                   "Yang termasuk metodologi pengembangan perangkat lunak: @ agile", "@ waterfall.",
                   "Sektor Unit"])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["range"]["tools"] == ["lembar kerja", "basis data catatan"]
    assert u["range"]["context"] == ["Register dapat mencakup: register kertas; register elektronik",
                                     "Situs aplikasi internet dapat mencakup: perbankan; belanja",
                                     "Yang termasuk metodologi pengembangan perangkat lunak: agile; waterfall"]


def test_after_a_sector_row_printed_on_the_english_page_the_indonesian_range_goes_on():
    # CUFPOS402A: "Sektor Unit" stood on the English page, and the Indonesian range
    # page after it was filed under that row and lost
    cut = OLDER.index("Range Statement")
    lines = OLDER[:cut] + [
        "Range Statement",
        "The range statement relates to the unit of competency as a whole.",
        "Registers may",
        "include:",
        "@ paper registers",
        "", "39", "",
        "Batasan Variabel",
        "Batasan variabel berkaitan dengan unit kompetensi secara keseluruhan.",
        "Register dapat",
        "mencakup:",
        "@ register kertas",
        "", "40", "",
        "@ electronic registers.",
        "Archives may include:",
        "@ boxes",
        "@ shelves.",
        "Unit Sector(s)",
        "Administration",
        "Sektor Unit",
        "Administrasi",
        "", "41", "",
        "@ register elektronik.",
        "Arsip dapat mencakup:",
        "@ kotak dari kertas",
        "@ rak.",
        "", "42", ""]
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["range"]["context"] == ["Register dapat mencakup: register kertas; register elektronik",
                                     "Arsip dapat mencakup: kotak dari kertas; rak"]


def test_a_page_of_names_takes_the_language_of_the_heading_further_down_it():
    # (found while repairing, ICAGAM504A) pages that hold only names have no function
    # words: the English one went into the Indonesian range, the Indonesian one was lost
    cut = OLDER.index("Range Statement")
    lines = OLDER[:cut] + [
        "Range Statement",
        "The range statement relates to the unit of competency as a whole.",
        "Communication may include:",
        "@ electronic",
        "@ forms",
        "", "39", "",
        "Batasan Variabel",
        "Batasan variabel berkaitan dengan unit kompetensi secara keseluruhan.",
        "Komunikasi dapat meliputi:",
        "@ elektronik",
        "@ formulir",
        "", "40", "",
        "@ post-it notes",
        "@ whiteboards.",
        "Unit Sector(s)",
        "Administration",
        "", "41", "",
        "@ catatan tempel",
        "@ papan tulis.",
        "Sektor Unit",
        "Administrasi",
        "", "42", ""]
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["range"]["context"] == ["Komunikasi dapat meliputi: elektronik; formulir; catatan tempel; papan tulis"]


def test_a_range_item_goes_on_inside_its_parenthesis():
    # (found while repairing, BSBSUS301A) "(seperti" / "Undang-Undang ..." / "...):"
    # started a variable inside the item
    lines = _swap(OLDER, ["Register dapat", "mencakup:", "@ register kertas", "@ register elektronik."],
                  ["Pemenuhan dapat", "mencakup:", "@ hukum yang relevan (seperti", "Undang-Undang Kearsipan",
                   "Nasional):", "@ tingkat nasional", "@ tingkat daerah."])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["range"]["context"] == ["Pemenuhan dapat mencakup: hukum yang relevan (seperti Undang-Undang "
                                     "Kearsipan Nasional):; tingkat nasional; tingkat daerah"]
    # once the parenthesis is closed, a variable can start again
    lines = _swap(OLDER, ["Register dapat", "mencakup:", "@ register kertas", "@ register elektronik."],
                  ["Arsip dapat", "mencakup:", "@ arsip lama (lihat", "Undang-Undang Kearsipan", "Nasional)",
                   "Rak dapat mencakup:", "@ rak besi."])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["range"]["context"] == ["Arsip dapat mencakup: arsip lama (lihat Undang-Undang Kearsipan Nasional)",
                                     "Rak dapat mencakup: rak besi"]


def test_bullets_inside_the_description_and_doubled_bullets_do_not_leak():
    # ICTTEN5201A: "ketika bekerja: ● instalasi ..." in the description, and
    # "● •Pengelola jaringan" (ICADBS602A: "• Lotus Notes") in a list
    lines = _swap(NEWER, ["Unit ini menjelaskan keterampilan dan pengetahuan yang diperlukan untuk",
                          "membangun widget contoh."],
                  ["Unit ini berlaku untuk petugas ketika bekerja:", "% merakit widget", "% menguji widget."])
    lines = _swap(lines, ["% membangun dua widget"], ["% •membangun dua widget"])
    (u,) = SK.parse_training_package(_pdf(lines))
    assert u["description"] == "Unit ini berlaku untuk petugas ketika bekerja: merakit widget; menguji widget."
    assert u["skills"][0] == "membangun dua widget"


def test_the_indonesian_title_is_the_one_of_the_decrees_list_of_units():
    # ICANWK501A: the unit's own heading is a rough translation; the decree lists the
    # official title (the list's last row must not take "B." of "B. Uraian ...")
    lines = [{"ICTXMP401 Membangun widget contoh": "ICTXMP401 Bangun contoh widget-widget"}.get(l, l)
             for l in NEWER]
    older, newer = SK.parse_training_package(_pdf(LIST_OF_UNITS + OLDER + lines))
    assert (older["title"], newer["title"]) == ("Memelihara Catatan Contoh", "Membangun Widget Contoh")
    assert newer["title_en"] == "Build sample widgets"            # the English one stays the heading's
    (alone,) = SK.parse_training_package(_pdf(lines))             # not in a list: its heading
    assert alone["title"] == "Bangun contoh widget-widget"
    rows = SK.training_package_titles(_pdf(LIST_OF_UNITS))
    assert rows == {"ICAXMP301A": ["Maintain Sample Records ", "Memelihara Catatan Contoh "],
                    "ICTXMP401": ["Build Sample Widgets ", "Membangun Widget Contoh "]}


def test_read_pdf_separates_pages_with_a_form_feed_line_which_parse_units_ignores():
    import types

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    fake = types.ModuleType("pypdf")
    fake.PdfReader = lambda path: types.SimpleNamespace(pages=[Page(" \n17 \nfirst "), Page("second "), Page(None)])
    saved = sys.modules.get("pypdf")
    sys.modules["pypdf"] = fake
    try:
        text = SK.read_pdf(Path("invented.pdf"))
    finally:
        if saved is None:
            sys.modules.pop("pypdf", None)
        else:
            sys.modules["pypdf"] = saved
    assert text == " \n17 \nfirst \n\f\nsecond \n\f\n"
    # the classic parser reads the form feed as an empty line: same units
    paged = SKKNI_TEXT.replace("\n12\n", "\n\f\n12\n").replace("\nBATASAN VARIABEL", "\n\f\nBATASAN VARIABEL")
    assert paged != SKKNI_TEXT and SK.parse_units(paged, source="x") == SK.parse_units(SKKNI_TEXT, source="x")


def _run_standalone() -> int:
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:                                            # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
