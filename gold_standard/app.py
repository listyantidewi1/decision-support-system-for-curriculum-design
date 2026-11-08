"""Streamlit span-annotation UI for the gold standard — the interactive delivery
format, equivalent to the Form A-1 workbook (export_xlsx.py).

Span capture: the annotator clicks the words of the sentence (chips) and the app
derives start_char / end_char exactly, snapped to token boundaries (which the
guideline's boundary rules want anyway). The selection is highlighted in place.
Each annotator works in their own file, blind to the others (guideline Bagian 10.1).

Run:  .venv-ml\\Scripts\\streamlit run gold_standard/app.py
Open with ?annotator=AV (or AI / AD), or pick in the sidebar.
Annotations autosave to DATA/gold_standard/annotations/<split>_<code>.json
and export to a Form A-1 CSV on demand.
"""
from __future__ import annotations

import html as _html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = REPO_ROOT / "DATA" / "gold_standard"
ANN_DIR = GS_DIR / "annotations"

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

# Condensed guideline (B-1) for the in-app reference expander.
TAG_HELP = {
    "SH": "Tindakan kerja teknis yang dapat dilakukan — mis. «mengembangkan REST API», «menulis kueri SQL».",
    "SS": "Kemampuan antarpribadi / mengelola diri — mis. «komunikasi», «kolaborasi».",
    "KN": "Konsep/prinsip yang perlu dipahami — mis. «object-oriented programming», «struktur data».",
    "TT": "Nama teknologi tertentu — mis. «Python», «Docker», «PostgreSQL».",
    "WB": "Sifat/watak dalam bekerja — mis. «teliti», «proaktif», «detail-oriented».",
}


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]


# Trim separator punctuation off span ends (guideline 6.7: punctuation stays out of
# the span), while keeping meaningful trailing chars like + or # (C++, C#).
_TRIM_CHARS = " \t\r\n,;:.•·"


def trim_span(sent: str, start: int, end: int) -> tuple[int, int]:
    while start < end and sent[start] in _TRIM_CHARS:
        start += 1
    while end > start and sent[end - 1] in _TRIM_CHARS:
        end -= 1
    return start, end


def esc(s: str) -> str:
    return _html.escape(s or "")


@st.cache_data
def load_sample(split: str) -> pd.DataFrame:
    return pd.read_parquet(GS_DIR / f"{split}.parquet")


def ann_path(split: str, code: str) -> Path:
    return ANN_DIR / f"{split}_{code}.json"


def load_ann(split: str, code: str) -> dict:
    p = ann_path(split, code)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


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
                         "end_char": "", "label": "", "confidence": "", "flag": "",
                         "note": a.get("note", "")})
        for i, s in enumerate(spans, 1):
            rows.append({**base, "span_no": i, "span_text": s["span_text"],
                         "start_char": s["start_char"], "end_char": s["end_char"],
                         "label": s["label"], "confidence": s["confidence"],
                         "flag": s.get("flag", ""), "note": s.get("note", "")})
    return pd.DataFrame(rows)


def highlight(sent: str, toks: list, sel: set[int]) -> str:
    out, prev = [], 0
    for i, (t, a, b) in enumerate(toks):
        out.append(esc(sent[prev:a]))
        chip = esc(t)
        out.append(f"<mark style='background:#ffd966;border-radius:3px'>{chip}</mark>"
                   if i in sel else chip)
        prev = b
    out.append(esc(sent[prev:]))
    return "".join(out)


# --------------------------------------------------------------------------- UI
st.set_page_config(page_title="Anotasi Gold Standard", layout="wide")
st.markdown(
    "<style>[data-testid='stToolbar'],#MainMenu,footer{visibility:hidden}"
    "mark{padding:0 1px}</style>", unsafe_allow_html=True)
split = "pilot_100"

qp = st.query_params.get("annotator", "")
default_idx = list(ANNOTATORS).index(qp) if qp in ANNOTATORS else 0
with st.sidebar:
    st.header("Anotator")
    code = st.selectbox("Kode", list(ANNOTATORS), index=default_idx,
                        format_func=lambda c: f"{c} — {ANNOTATORS[c]}")
    st.caption(f"Pedoman AG-RPL-LIMA · data: {split}")

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
    if st.button("Ekspor Form A-1 (CSV)", use_container_width=True):
        csv = to_form_a1(sample, ann, code).to_csv(index=False).encode("utf-8")
        st.download_button("Unduh CSV", csv, file_name=f"{split}_{code}_form_a1.csv",
                           use_container_width=True)
    st.download_button("Unduh anotasi (JSON)",
                       json.dumps(ann, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"{split}_{code}.json", use_container_width=True)

i = st.session_state.idx
rec = sample.iloc[i]
sid = rec["sentence_id"]
cur = ann.get(sid, {})
toks = tokenize(rec["sentence_text"])
sel_key = f"pills::{sid}"
sel = sorted(st.session_state.get(sel_key, []) or [])
sel_set = set(sel)

with st.expander("📖 Pedoman ringkas (lima tag, urutan keputusan, batas span)"):
    for c, full in TAGS.items():
        st.markdown(f"**{c} · {full}** — {TAG_HELP[c]}")
    st.markdown("**Urutan keputusan:** 1) nama teknologi? → TT · 2) tindakan teknis? "
                "→ SH · 3) konsep yang dipahami? → KN · 4) komunikasi/kerja sama? → SS "
                "· 5) sifat/watak? → WB.")
    st.markdown("**Batas span:** ambil span terpendek yang tetap bermakna; jangan "
                "sertakan pembuka (must, able to, experience in); pisahkan tindakan "
                "dari teknologi; tanda baca di luar span. Pendidikan/pengalaman/"
                "sertifikasi/syarat bahasa → bukan tag (penanda OS).")

st.subheader(f"Kalimat {i + 1} / {len(sample)}")
st.caption(f"{rec['country_iso']} · {rec['site']} · bahasa {rec['posting_language']} "
           f"· {rec['quarter']} · {sid}")
ctx = []
if rec["previous_sentence"]:
    ctx.append(f"<span style='color:#999'>… {esc(rec['previous_sentence'])}</span>")
ctx.append(f"<div style='font-size:1.3rem;font-weight:600;line-height:1.6;margin:"
           f".4rem 0'>{highlight(rec['sentence_text'], toks, sel_set)}</div>")
if rec["next_sentence"]:
    ctx.append(f"<span style='color:#999'>{esc(rec['next_sentence'])} …</span>")
st.markdown("".join(ctx), unsafe_allow_html=True)
st.divider()

c1, c2 = st.columns(2)
rel = c1.radio("Relevansi kalimat", RELEVANCE,
               index=RELEVANCE.index(cur["relevance"]) if cur.get("relevance") else 0)
has5 = c2.radio("has_5tag (minimal satu dari lima tag?)", HAS_5TAG,
                index=HAS_5TAG.index(cur["has_5tag"]) if cur.get("has_5tag") else 1,
                horizontal=True)

st.markdown("#### Tandai span")
st.pills("Klik kata-kata entitas (berdampingan)", options=list(range(len(toks))),
         format_func=lambda n: toks[n][0], selection_mode="multi", key=sel_key)
span_text, span_start, span_end = "", 0, 0
if sel:
    if sel != list(range(sel[0], sel[-1] + 1)):
        st.warning("Kata tidak berdampingan — span diambil dari kata pertama s.d. terakhir.")
    span_start, span_end = trim_span(
        rec["sentence_text"], toks[sel[0]][1], toks[sel[-1]][2])
    span_text = rec["sentence_text"][span_start:span_end]
    st.success(f"Span terpilih: «{span_text}»  ({span_start}:{span_end})")

s1, s2, s3 = st.columns(3)
label = s1.selectbox("Label", [""] + list(TAGS),
                     format_func=lambda c: "— pilih label —" if c == "" else f"{c} — {TAGS[c]}")
conf = s2.selectbox("Keyakinan", CONFIDENCE,
                    format_func=lambda x: f"{x} — {'sangat yakin' if x == 4 else 'yakin' if x == 3 else 'kurang yakin' if x == 2 else 'tidak yakin'}")
flag = s3.selectbox("Penanda", list(FLAGS), format_func=lambda c: FLAGS[c])
note = st.text_input("Catatan (wajib bila ada penanda / penyebutan negatif / keyakinan ≤ 2)")

add_disabled = not sel or label == ""
if st.button("➕ Tambah span", disabled=add_disabled, type="primary"):
    cur.setdefault("spans", []).append(
        {"span_text": span_text, "start_char": span_start, "end_char": span_end,
         "label": label, "confidence": conf, "flag": flag, "note": note})
    cur["relevance"], cur["has_5tag"] = rel, has5
    ann[sid] = cur
    save_ann(split, code, ann)
    st.session_state.pop(sel_key, None)      # clear chip selection
    st.rerun()
if sel and label == "":
    st.caption("Pilih label dulu untuk menambah span.")

if cur.get("spans"):
    st.markdown("**Span pada kalimat ini:**")
for n, s in enumerate(cur.get("spans", [])):
    cols = st.columns([6, 3, 2, 1])
    cols[0].markdown(f"«{esc(s['span_text'])}»")
    cols[1].write(f"{s['label']} · conf {s['confidence']}"
                  + (f" · {s['flag']}" if s.get("flag") else ""))
    cols[2].caption(f"[{s['start_char']}:{s['end_char']}]")
    if cols[3].button("🗑", key=f"del{sid}{n}"):
        cur["spans"].pop(n)
        ann[sid] = cur
        save_ann(split, code, ann)
        st.rerun()

st.divider()
n1, n2, n3 = st.columns(3)


def _persist():
    cur["relevance"], cur["has_5tag"] = rel, has5
    ann[sid] = cur
    save_ann(split, code, ann)


if n1.button("⟵ Sebelumnya", disabled=i == 0, use_container_width=True):
    _persist()
    st.session_state.idx = max(0, i - 1)
    st.rerun()
if n2.button("💾 Simpan kalimat", use_container_width=True):
    _persist()
    st.success("Tersimpan.")
if n3.button("Berikutnya ⟶", disabled=i >= len(sample) - 1, use_container_width=True):
    _persist()
    st.session_state.idx = min(len(sample) - 1, i + 1)
    st.rerun()
