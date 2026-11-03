"""Streamlit span-annotation UI for the gold standard — the interactive delivery
format, equivalent to the Form A-1 workbook (export_xlsx.py).

Edge over the spreadsheet: the annotator selects whole tokens of the sentence and
the app derives start_char / end_char exactly (snapped to token boundaries, which
the guideline's boundary rules want anyway — punctuation excluded, shortest span).
Each annotator works in their own file, blind to the others (guideline Bagian 10.1).

Run:  .venv-ml\\Scripts\\streamlit run gold_standard/app.py
Open with ?annotator=AV (or AI / AD), or pick in the sidebar.
Annotations autosave to DATA/gold_standard/annotations/<split>_<code>.json
and export to a Form A-1 CSV on demand.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
ANN_DIR = GS_DIR / "annotations"

# Vocabulary (mirrors gold_standard/__init__.py; kept local so the app can run
# standalone via `streamlit run` without the package import path).
TAGS = {"SH": "SKILL_HARD", "SS": "SKILL_SOFT", "KN": "KNOWLEDGE",
        "TT": "TOOL_TECHNOLOGY", "WB": "WORK_BEHAVIOR"}
FLAGS = {"": "(tidak ada)", "AL": "AMBIGUOUS_LABEL", "AB": "AMBIGUOUS_BOUNDARY",
         "IC": "INSUFFICIENT_CONTEXT", "OS": "OUT_OF_SCHEMA",
         "LU": "LANGUAGE_UNCERTAIN", "PD": "POSSIBLE_DUPLICATE",
         "GG": "GUIDELINE_GAP", "OP": "OVERLAP_POTENTIAL"}
RELEVANCE = ["RELEVANT", "NON_RELEVANT", "UNCERTAIN"]
HAS_5TAG = ["YA", "TIDAK"]
CONFIDENCE = [4, 3, 2, 1]
ANNOTATORS = {"AV": "Ahli Pendidikan Vokasi", "AI": "Ahli / praktisi industri RPL",
              "AD": "Ahli sains data / NLP"}
_TOKEN_RE = re.compile(r"\S+")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]


@st.cache_data
def load_sample(split: str) -> pd.DataFrame:
    return pd.read_parquet(GS_DIR / f"{split}.parquet")


def ann_path(split: str, code: str) -> Path:
    return ANN_DIR / f"{split}_{code}.json"


def load_ann(split: str, code: str) -> dict:
    p = ann_path(split, code)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_ann(split: str, code: str, data: dict) -> None:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    data["_meta"] = {"annotator": code, "split": split,
                     "saved_utc": datetime.now(timezone.utc).isoformat()}
    ann_path(split, code).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def to_form_a1(sample: pd.DataFrame, ann: dict, code: str) -> pd.DataFrame:
    rows = []
    for rec in sample.itertuples(index=False):
        a = ann.get(rec.sentence_id, {})
        spans = a.get("spans", [])
        base = dict(annotator=code, sentence_id=rec.sentence_id,
                    posting_id=rec.posting_id, country_iso=rec.country_iso,
                    site=rec.site, language=rec.posting_language,
                    sentence_text=rec.sentence_text,
                    relevance=a.get("relevance", ""), has_5tag=a.get("has_5tag", ""))
        if not spans:
            rows.append({**base, "span_no": "", "span_text": "", "start_char": "",
                         "end_char": "", "label": "", "confidence": "",
                         "flag": "", "note": a.get("note", "")})
        for i, s in enumerate(spans, 1):
            rows.append({**base, "span_no": i, "span_text": s["span_text"],
                         "start_char": s["start_char"], "end_char": s["end_char"],
                         "label": s["label"], "confidence": s["confidence"],
                         "flag": s.get("flag", ""), "note": s.get("note", "")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- UI
st.set_page_config(page_title="Anotasi Gold Standard", layout="wide")
split = "pilot_100"

qp = st.query_params.get("annotator", "")
default_idx = (list(ANNOTATORS).index(qp) if qp in ANNOTATORS else 0)
with st.sidebar:
    st.header("Anotator")
    code = st.selectbox("Kode", list(ANNOTATORS), index=default_idx,
                        format_func=lambda c: f"{c} — {ANNOTATORS[c]}")
    st.caption(f"Pedoman AG-RPL-LIMA · berkas data: {split}")

try:
    sample = load_sample(split)
except FileNotFoundError:
    st.error(f"Sample {split} belum dibuat. Jalankan: python -m gold_standard.sample")
    st.stop()

if "ann" not in st.session_state or st.session_state.get("ann_code") != code:
    st.session_state.ann = load_ann(split, code)
    st.session_state.ann_code = code
    st.session_state.idx = 0
ann = st.session_state.ann

done = sum(1 for sid in sample["sentence_id"] if ann.get(sid, {}).get("relevance"))
with st.sidebar:
    st.progress(done / len(sample), text=f"{done}/{len(sample)} kalimat dinilai")
    jump = st.number_input("Ke kalimat #", 1, len(sample), st.session_state.idx + 1)
    st.session_state.idx = int(jump) - 1
    st.divider()
    if st.button("Ekspor Form A-1 (CSV)"):
        csv = to_form_a1(sample, ann, code).to_csv(index=False).encode("utf-8")
        st.download_button("Unduh CSV", csv, file_name=f"{split}_{code}_form_a1.csv")
    st.download_button("Unduh anotasi (JSON)",
                       json.dumps(ann, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"{split}_{code}.json")

i = st.session_state.idx
rec = sample.iloc[i]
sid = rec["sentence_id"]
cur = ann.get(sid, {})

st.subheader(f"Kalimat {i + 1} / {len(sample)}")
st.caption(f"{rec['country_iso']} · {rec['site']} · bahasa {rec['posting_language']} "
           f"· {rec['quarter']} · {sid}")
if rec["previous_sentence"]:
    st.markdown(f"<span style='color:gray'>… {rec['previous_sentence']}</span>",
                unsafe_allow_html=True)
st.markdown(f"### {rec['sentence_text']}")
if rec["next_sentence"]:
    st.markdown(f"<span style='color:gray'>{rec['next_sentence']} …</span>",
                unsafe_allow_html=True)
st.divider()

c1, c2 = st.columns(2)
rel = c1.radio("Relevansi kalimat", RELEVANCE,
               index=RELEVANCE.index(cur["relevance"]) if cur.get("relevance") else 0)
has5 = c2.radio("has_5tag (minimal satu dari lima tag?)", HAS_5TAG,
                index=HAS_5TAG.index(cur["has_5tag"]) if cur.get("has_5tag") else 1,
                horizontal=True)

st.markdown("#### Span")
toks = tokenize(rec["sentence_text"])
opts = [f"{n}:{t[0]}" for n, t in enumerate(toks)]
sel = st.multiselect("Pilih token entitas (berurutan)", opts,
                     help="Pilih token yang berdampingan; offset dihitung otomatis.")
span_preview = ""
if sel:
    idxs = sorted(int(o.split(":", 1)[0]) for o in sel)
    lo, hi = idxs[0], idxs[-1]
    if idxs != list(range(lo, hi + 1)):
        st.warning("Token tidak berurutan — span diambil dari token pertama s.d. terakhir.")
    start, end = toks[lo][1], toks[hi][2]
    span_preview = rec["sentence_text"][start:end]
    st.info(f"Span: «{span_preview}»  (start={start}, end={end})")

s1, s2, s3 = st.columns(3)
label = s1.selectbox("Label", list(TAGS), format_func=lambda c: f"{c} — {TAGS[c]}")
conf = s2.selectbox("Keyakinan", CONFIDENCE)
flag = s3.selectbox("Penanda", list(FLAGS), format_func=lambda c: FLAGS[c])
note = st.text_input("Catatan (wajib bila ada penanda / penyebutan negatif / keyakinan ≤ 2)")

if st.button("➕ Tambah span", disabled=not sel):
    idxs = sorted(int(o.split(":", 1)[0]) for o in sel)
    start, end = toks[idxs[0]][1], toks[idxs[-1]][2]
    cur.setdefault("spans", []).append(
        {"span_text": rec["sentence_text"][start:end], "start_char": start,
         "end_char": end, "label": label, "confidence": conf, "flag": flag,
         "note": note})
    cur["relevance"], cur["has_5tag"] = rel, has5
    ann[sid] = cur
    save_ann(split, code, ann)
    st.rerun()

for n, s in enumerate(cur.get("spans", [])):
    cols = st.columns([6, 2, 1, 1, 1])
    cols[0].write(f"«{s['span_text']}»")
    cols[1].write(f"{s['label']} · conf {s['confidence']}" + (f" · {s['flag']}" if s.get("flag") else ""))
    cols[2].write(f"[{s['start_char']}:{s['end_char']}]")
    if cols[4].button("🗑", key=f"del{sid}{n}"):
        cur["spans"].pop(n)
        ann[sid] = cur
        save_ann(split, code, ann)
        st.rerun()

st.divider()
n1, n2, n3 = st.columns(3)
if n1.button("⟵ Sebelumnya", disabled=i == 0):
    cur["relevance"], cur["has_5tag"] = rel, has5
    ann[sid] = cur
    save_ann(split, code, ann)
    st.session_state.idx = max(0, i - 1)
    st.rerun()
if n2.button("💾 Simpan kalimat ini"):
    cur["relevance"], cur["has_5tag"] = rel, has5
    ann.setdefault(sid, cur)
    ann[sid] = cur
    save_ann(split, code, ann)
    st.success("Tersimpan.")
if n3.button("Berikutnya ⟶", disabled=i >= len(sample) - 1):
    cur["relevance"], cur["has_5tag"] = rel, has5
    ann[sid] = cur
    save_ann(split, code, ann)
    st.session_state.idx = min(len(sample) - 1, i + 1)
    st.rerun()
