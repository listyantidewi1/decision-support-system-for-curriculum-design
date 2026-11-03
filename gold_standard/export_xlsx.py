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
SPAN_ROWS_PER_SENTENCE = 6

HEADERS = ["row_type", "sentence_id", "meta", "previous_sentence", "sentence_text",
           "next_sentence", "relevance", "has_5tag", "span_no", "span_text",
           "start_char", "end_char", "label", "confidence", "flag",
           "parent_span_id", "note"]

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
         "satu baris SPAN di bawahnya: salin teks span persis dari kalimat, pilih "
         "label dan tingkat keyakinan, dan isi penanda bila ada. start_char/end_char "
         "boleh dikosongkan — diisi oleh sistem."),
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


def _anotasi(ws, sample: pd.DataFrame) -> None:
    widths = {"A": 10, "B": 20, "C": 16, "D": 34, "E": 46, "F": 34, "G": 15,
              "H": 9, "I": 7, "J": 30, "K": 9, "L": 9, "M": 8, "N": 11,
              "O": 7, "P": 14, "Q": 28}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for c, h in enumerate(HEADERS, start=1):
        cell = ws.cell(1, c, h)
        cell.fill, cell.font, cell.alignment, cell.border = (
            _HEAD_FILL, _HEAD_FONT, _TOP, _BORDER)
    ws.freeze_panes = "A2"

    dv_rel, dv_has = _dv(RELEVANCE, False), _dv(HAS_5TAG, False)
    dv_label = _dv(list(TAGS), True)
    dv_conf = _dv([str(x) for x in CONFIDENCE], True)
    dv_flag = _dv(list(FLAGS), True)
    for dv in (dv_rel, dv_has, dv_label, dv_conf, dv_flag):
        ws.add_data_validation(dv)

    row = 2
    for rec in sample.itertuples(index=False):
        meta = f"{rec.country_iso}/{rec.site}/{rec.posting_language} · {rec.quarter}"
        ws.cell(row, 1, "SENTENCE")
        ws.cell(row, 2, rec.sentence_id)
        ws.cell(row, 3, meta)
        ws.cell(row, 4, rec.previous_sentence)
        ws.cell(row, 5, rec.sentence_text)
        ws.cell(row, 6, rec.next_sentence)
        for c in range(1, len(HEADERS) + 1):
            cc = ws.cell(row, c)
            cc.fill, cc.border = _SENTENCE_FILL, _BORDER
            cc.alignment = _WRAP if c in (4, 5, 6) else _TOP
        dv_rel.add(ws.cell(row, 7))
        dv_has.add(ws.cell(row, 8))
        sent_row = row
        row += 1
        for i in range(1, SPAN_ROWS_PER_SENTENCE + 1):
            ws.cell(row, 1, "SPAN")
            ws.cell(row, 2, rec.sentence_id)
            ws.cell(row, 9, i)
            for c in range(1, len(HEADERS) + 1):
                cc = ws.cell(row, c)
                cc.fill, cc.border = _SPAN_FILL, _BORDER
                cc.alignment = _WRAP if c == 10 else _TOP
            dv_label.add(ws.cell(row, 13))
            dv_conf.add(ws.cell(row, 14))
            dv_flag.add(ws.cell(row, 15))
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
