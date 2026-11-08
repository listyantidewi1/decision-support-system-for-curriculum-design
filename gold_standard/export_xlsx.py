"""Deliver a sentence sample as a Form A-1 annotation workbook (one per annotator).

Produces, under DATA/gold_standard/worksheets/, one .xlsx per annotator (AV/AI/AD)
plus a blank MASTER template. Each workbook has:
  * "Panduan" — Form A-0 identity + independence statement, the five-tag / flag /
    confidence / relevance legends, and the pre-save checklist (guideline Bagian 11);
  * "Anotasi" — the sample in Form A-1 layout: one SENTENCE row per sentence
    (relevance + has_5tag dropdowns) followed by blank SPAN rows (label / confidence
    / flag dropdowns). start_char/end_char are optional — the researcher derives them
    from span_text, so annotators only type the span text, pick a label and a
    confidence, and add a flag when needed.

This is the spreadsheet delivery format; app.py is the equivalent Streamlit one.

Run:  .venv-ml\\Scripts\\python.exe -m gold_standard.export_xlsx [--split pilot_100]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from governance.ladder import REPO_ROOT
from . import (ANNOTATORS, CONFIDENCE, FLAGS, GUIDELINE_CODE, HAS_5TAG,
               RELEVANCE, TAGS)

GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
OUT_DIR = GS_DIR / "worksheets"
# Generous prepared SPAN rows per sentence so even enumeration-dense sentences don't
# overflow; annotators may still insert more rows (whole-column validation covers them).
SPAN_ROWS_PER_SENTENCE = 14

# occurrence_no disambiguates a span_text that appears more than once in the sentence
# (default 1 = first match), so the researcher can recover exact offsets the way the
# Streamlit app captures them directly.
HEADERS = ["row_type", "sentence_id", "meta", "previous_sentence", "sentence_text",
           "next_sentence", "relevance", "has_5tag", "span_no", "span_text",
           "occurrence_no", "start_char", "end_char", "label", "confidence", "flag",
           "parent_span_id", "note"]
COL = {h: i + 1 for i, h in enumerate(HEADERS)}

_SENTENCE_FILL = PatternFill("solid", fgColor="DDEBF7")
_SPAN_FILL = PatternFill("solid", fgColor="F2F2F2")
_HEAD_FILL = PatternFill("solid", fgColor="1F4E78")
_HEAD_FONT = Font(color="FFFFFF", bold=True, size=10)
_THIN = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_WRAP = Alignment(wrap_text=True, vertical="top")
_TOP = Alignment(vertical="top")


def _dv(options: list[str], allow_blank: bool = True) -> DataValidation:
    formula = '"' + ",".join(str(o) for o in options) + '"'
    return DataValidation(type="list", formula1=formula, allow_blank=allow_blank,
                          showDropDown=False)


def _panduan(ws, code: str) -> None:
    name = ANNOTATORS.get(code, "(templat kosong — salin untuk tiap anotator)")
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 80
    rows = [
        ("Pedoman", GUIDELINE_CODE),
        ("Lembar anotasi (Formulir A-0 / A-1)", f"Anotator {code} — {name}"),
        ("", ""),
        ("Kode anotator", code),
        ("Nama lengkap", ""),
        ("Bidang keahlian", name),
        ("Institusi", ""),
        ("Tanggal kalibrasi / anotasi", ""),
        ("", ""),
        ("Pernyataan independensi",
         "Saya menganotasi secara independen, tidak melihat anotasi anotator lain "
         "maupun keluaran model sebelum anotasi saya selesai, dan melaporkan setiap "
         "kasus yang tidak diatur dalam pedoman."),
        ("Tanda tangan", ""),
        ("", ""),
        ("CARA MENGISI", "Pada sheet 'Anotasi': untuk tiap kalimat, isi baris "
         "SENTENCE (relevansi + has_5tag). Untuk tiap entitas yang tertulis, isi "
         "satu baris SPAN di bawahnya: salin teks span PERSIS dari kalimat, pilih "
         "label dan tingkat keyakinan, dan isi penanda bila ada. start_char/end_char "
         "diisi oleh sistem. Jika teks span muncul lebih dari sekali dalam kalimat, "
         "isi occurrence_no dengan kemunculan ke berapa (1 = pertama). Bila entitas "
         "lebih banyak daripada baris SPAN yang tersedia, sisipkan baris SPAN baru di "
         "bawahnya (daftar dropdown tetap berlaku)."),
        ("", ""),
        ("LABEL (lima tag)", ""),
    ]
    for code_, label_ in TAGS.items():
        rows.append((f"  {code_}", label_))
    rows += [("", ""), ("PENANDA (flag)", "")]
    for code_, name_ in FLAGS.items():
        rows.append((f"  {code_}", name_))
    rows += [
        ("", ""),
        ("KEYAKINAN", "4 sangat yakin · 3 yakin · 2 kurang yakin · 1 tidak yakin"),
        ("RELEVANSI", " / ".join(RELEVANCE)),
        ("has_5tag", " / ".join(HAS_5TAG) + "  (minimal satu dari lima tag?)"),
        ("", ""),
        ("PEMERIKSAAN SEBELUM MENYIMPAN", ""),
    ]
    checklist = [
        "Relevansi kalimat telah ditentukan.",
        "Penanda has_5tag telah diisi.",
        "Seluruh entitas yang benar-benar tertulis telah diperiksa.",
        "Tidak ada entitas yang hanya berupa simpulan anotator.",
        "Nama teknologi telah dipisahkan dari tindakan kerja bila memungkinkan.",
        "Label SH/KN dan SS/WB tidak tertukar.",
        "Batas span sesingkat mungkin tetapi tetap bermakna.",
        "Tanda baca tidak masuk ke dalam span tanpa alasan.",
        "Teks span sama persis dengan teks sumber.",
        "Kasus bermakna ganda telah diberi penanda dan catatan.",
        "Tingkat keyakinan telah diisi untuk setiap span.",
    ]
    for c in checklist:
        rows.append(("  ☐", c))

    for r, (a, b) in enumerate(rows, start=1):
        ws.cell(r, 1, a).font = Font(bold=a.isupper() and bool(a))
        cell = ws.cell(r, 2, b)
        cell.alignment = _WRAP
    ws.cell(1, 1).font = Font(bold=True, size=13)


_WRAP_COLS = {COL["previous_sentence"], COL["sentence_text"], COL["next_sentence"]}
_MAXROW = 5000   # whole-column validation range, covers any rows annotators insert


def _anotasi(ws, sample: pd.DataFrame) -> None:
    widths = {"row_type": 10, "sentence_id": 20, "meta": 16, "previous_sentence": 30,
              "sentence_text": 46, "next_sentence": 30, "relevance": 15, "has_5tag": 9,
              "span_no": 7, "span_text": 30, "occurrence_no": 8, "start_char": 9,
              "end_char": 9, "label": 15, "confidence": 13, "flag": 14,
              "parent_span_id": 12, "note": 26}
    for name, w in widths.items():
        ws.column_dimensions[get_column_letter(COL[name])].width = w
    for c, h in enumerate(HEADERS, start=1):
        cell = ws.cell(1, c, h)
        cell.fill, cell.font, cell.alignment, cell.border = (
            _HEAD_FILL, _HEAD_FONT, _TOP, _BORDER)
    ws.freeze_panes = "A2"

    # Whole-column data validations so any inserted row inherits the pick-list too.
    def _col_range(name: str) -> str:
        L = get_column_letter(COL[name])
        return f"{L}2:{L}{_MAXROW}"

    for name, options in (("relevance", RELEVANCE), ("has_5tag", HAS_5TAG),
                          ("label", list(TAGS)), ("flag", list(FLAGS)),
                          ("confidence", [str(x) for x in CONFIDENCE])):
        dv = _dv(options, True)
        dv.add(_col_range(name))
        ws.add_data_validation(dv)

    row = 2
    for rec in sample.itertuples(index=False):
        meta = f"{rec.country_iso}/{rec.site}/{rec.posting_language} · {rec.quarter}"
        ws.cell(row, COL["row_type"], "SENTENCE")
        ws.cell(row, COL["sentence_id"], rec.sentence_id)
        ws.cell(row, COL["meta"], meta)
        ws.cell(row, COL["previous_sentence"], rec.previous_sentence)
        ws.cell(row, COL["sentence_text"], rec.sentence_text)
        ws.cell(row, COL["next_sentence"], rec.next_sentence)
        for c in range(1, len(HEADERS) + 1):
            cc = ws.cell(row, c)
            cc.fill, cc.border = _SENTENCE_FILL, _BORDER
            cc.alignment = _WRAP if c in _WRAP_COLS else _TOP
        sent_row = row
        row += 1
        for i in range(1, SPAN_ROWS_PER_SENTENCE + 1):
            ws.cell(row, COL["row_type"], "SPAN")
            ws.cell(row, COL["sentence_id"], rec.sentence_id)
            ws.cell(row, COL["span_no"], i)
            ws.cell(row, COL["occurrence_no"], 1)   # default to the 1st match
            for c in range(1, len(HEADERS) + 1):
                cc = ws.cell(row, c)
                cc.fill, cc.border = _SPAN_FILL, _BORDER
                cc.alignment = _WRAP if c == COL["span_text"] else _TOP
            row += 1
        ws.row_dimensions[sent_row].height = 46


def build_workbook(sample: pd.DataFrame, code: str, out: Path) -> None:
    wb = Workbook()
    _panduan(wb.active, code)
    wb.active.title = "Panduan"
    _anotasi(wb.create_sheet("Anotasi"), sample)
    wb.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="pilot_100")
    args = ap.parse_args()
    sample = pd.read_parquet(GS_DIR / f"{args.split}.parquet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for code in list(ANNOTATORS) + ["MASTER"]:
        out = OUT_DIR / f"{args.split}_{code}.xlsx"
        build_workbook(sample, code, out)
        print(f"-> {out}", flush=True)
    print(f"{len(sample)} sentences x {SPAN_ROWS_PER_SENTENCE} span rows each; "
          f"{len(ANNOTATORS)} annotator workbooks + MASTER template", flush=True)


if __name__ == "__main__":
    main()
