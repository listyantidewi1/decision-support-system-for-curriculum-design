"""Streamlit span-annotation UI for the gold standard — the interactive delivery
format, equivalent to the Form A-1 workbook (export_xlsx.py).

Span capture: the annotator clicks the words of the sentence (chips) and the app
derives start_char / end_char exactly, snapped to token boundaries (which the
guideline's boundary rules want anyway). The selection is highlighted in place.
Each annotator works in their own file, locked to their ?annotator= link and blind
to the others (guideline Bagian 10.1).

State model (important): every per-sentence widget is keyed by (annotator, sentence)
and every per-span widget by a nonce, so nothing bleeds across sentences or
annotators, and an un-judged sentence is never autosaved with a default value.

Run:  .venv-ml\\Scripts\\streamlit run gold_standard/app.py
Open with ?annotator=AV (or AI / AD).
Annotations autosave to DATA/gold_standard/annotations/<split>_<code>.json
and export to a Form A-1 CSV on demand.
"""
from __future__ import annotations

import html as _html
import json
import os
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
CONF_WORD = {4: "sangat yakin", 3: "yakin", 2: "kurang yakin", 1: "tidak yakin"}
ANNOTATORS = {"AV": "Ahli Pendidikan Vokasi", "AI": "Ahli / praktisi industri RPL",
              "AD": "Ahli sains data / NLP"}
_TOKEN_RE = re.compile(r"\S+")

TAG_HELP = {
    "SH": "Tindakan kerja teknis yang dapat dilakukan — mis. «mengembangkan REST API», «menulis kueri SQL».",
    "SS": "Kemampuan antarpribadi / mengelola diri — mis. «komunikasi», «kolaborasi».",
    "KN": "Konsep/prinsip yang perlu dipahami — mis. «object-oriented programming», «struktur data».",
    "TT": "Nama teknologi tertentu — mis. «Python», «Docker», «PostgreSQL».",
    "WB": "Sifat/watak dalam bekerja — mis. «teliti», «proaktif», «detail-oriented».",
}

# Separator punctuation trimmed off span ends (rule 6.7). A leading '.' is NOT
# trimmed, so technology tokens like ".NET"/".htaccess" keep their dot; trailing
# sentence punctuation still goes. '*' is trimmed as a safety net for any markdown
# residue that reaches a token.
_TRIM_LEADING = " \t\r\n,;:*•·"
_TRIM_TRAILING = " \t\r\n,;:.*•·"


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]


def esc(s: str) -> str:
    return _html.escape(s or "")


def trim_span(sent: str, start: int, end: int) -> tuple[int, int]:
    while start < end and sent[start] in _TRIM_LEADING:
        start += 1
    while end > start and sent[end - 1] in _TRIM_TRAILING:
        end -= 1
    return start, end


@st.cache_data
def load_sample(split: str) -> pd.DataFrame:
    return pd.read_parquet(GS_DIR / f"{split}.parquet")


def ann_path(split: str, code: str) -> Path:
    return ANN_DIR / f"{split}_{code}.json"


# --------------------------------------------------------------- persistence
# Two backends, same API. Local files for solo/offline work; the collector
# service when GS_API_URL is set, which is required for any hosted deployment
# because every free host has an EPHEMERAL filesystem — local autosaves there
# are silently lost on restart. The collector also scopes each annotator to
# their own token, replacing the guessable ?annotator= parameter.
API_URL = os.environ.get("GS_API_URL", "").rstrip("/")


def _api_token() -> str:
    return st.query_params.get("token", "") or os.environ.get("GS_API_TOKEN", "")


def api_enabled() -> bool:
    return bool(API_URL and _api_token())


def _api(method: str, path: str, **kw):
    import requests
    r = requests.request(method, f"{API_URL}{path}", timeout=20,
                         headers={"Authorization": f"Bearer {_api_token()}"}, **kw)
    r.raise_for_status()
    return r.json()


def api_whoami() -> dict:
    """Resolve the token to (annotator, split) server-side. The client never
    chooses its own identity, so one annotator cannot open another's workspace."""
    return _api("GET", "/me")


def load_ann(split: str, code: str) -> dict:
    if api_enabled():
        return _api("GET", "/annotations")
    p = ann_path(split, code)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_ann(split: str, code: str, data: dict) -> None:
    """Local mode writes the whole file; API mode upserts only the sentences that
    changed, so a refresh or a second tab cannot clobber earlier work."""
    if api_enabled():
        for sid, rec in data.items():
            if sid == "_meta" or not isinstance(rec, dict):
                continue
            if st.session_state.get("_synced", {}).get(sid) != rec:
                _api("PUT", f"/annotations/{sid}", json=rec)
                st.session_state.setdefault("_synced", {})[sid] = json.loads(
                    json.dumps(rec))          # deep copy for later comparison
        return
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    data["_meta"] = {"annotator": code, "split": split,
                     "saved_utc": datetime.now(timezone.utc).isoformat()}
    ann_path(split, code).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# Canonical Form A-1 column order shared with export_xlsx.py (one row per span;
# a span-less sentence emits one row with empty span fields).
FORM_A1_COLUMNS = ["annotator", "sentence_id", "posting_id", "country_iso", "site",
                   "language", "sentence_text", "relevance", "has_5tag", "span_no",
                   "span_text", "start_char", "end_char", "label", "confidence",
                   "flag", "parent_span_id", "note"]


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
                         "flag": a.get("flag", ""), "parent_span_id": "",
                         "note": a.get("note", "")})
        for i, s in enumerate(spans, 1):
            rows.append({**base, "span_no": i, "span_text": s["span_text"],
                         "start_char": s["start_char"], "end_char": s["end_char"],
                         "label": s["label"], "confidence": s["confidence"],
                         "flag": s.get("flag", ""),
                         "parent_span_id": s.get("parent_span_id", ""),
                         "note": s.get("note", "")})
    return pd.DataFrame(rows, columns=FORM_A1_COLUMNS)


def committed_token_idxs(toks: list, spans: list) -> set[int]:
    out = set()
    for s in spans:
        for i, (_t, a, b) in enumerate(toks):
            if a < s["end_char"] and b > s["start_char"]:
                out.add(i)
    return out


def highlight(sent: str, toks: list, sel: set[int], committed: set[int]) -> str:
    out, prev = [], 0
    for i, (t, a, b) in enumerate(toks):
        out.append(esc(sent[prev:a]))
        chip = esc(t)
        if i in sel:                       # in-progress selection -> yellow
            chip = f"<mark style='background:#ffd966;border-radius:3px'>{chip}</mark>"
        elif i in committed:               # already-saved span -> green
            chip = f"<mark style='background:#b6e3b6;border-radius:3px'>{chip}</mark>"
        out.append(chip)
        prev = b
    out.append(esc(sent[prev:]))
    return "".join(out)


# --------------------------------------------------------------------------- UI
st.set_page_config(page_title="Anotasi Gold Standard", layout="wide")
st.markdown(
    "<style>[data-testid='stToolbar'],#MainMenu,footer{visibility:hidden}"
    "mark{padding:0 1px}</style>", unsafe_allow_html=True)
split = "pilot_100"

# Annotator identity: locked to the ?annotator= link so one annotator cannot open
# another's file (independence, Bagian 10.1). Free selection only without a param.
qp = st.query_params.get("annotator", "")
with st.sidebar:
    st.header("Anotator")
    if qp in ANNOTATORS:
        code = qp
        st.markdown(f"**{code} — {ANNOTATORS[code]}**")
    else:
        code = st.selectbox("Kode", list(ANNOTATORS),
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
st.session_state.setdefault("pills_nonce", 0)


def persist_sid(psid: str) -> None:
    """Save sentence-level answers for `psid`, reading the keyed widget state.
    Writes a record ONLY if the sentence was actually judged (relevance set), so
    navigating past an untouched sentence never fabricates a label.

    has_5tag is stored EXACTLY as the annotator answered it — never derived from
    the spans, never defaulted. It used to be forced to 'YA' whenever a span
    existed, forced to 'TIDAK' for NON_RELEVANT, and defaulted to 'TIDAK' when
    the radio was untouched. That made the has_5tag agreement statistic largely an
    artefact of this UI rather than three independent expert judgements, so no
    reliability figure could be claimed for it. An unanswered radio is now stored
    blank, and the IAA step treats blank as 'not answered' (it cannot count as
    agreement) and routes the sentence to adjudication."""
    r = st.session_state.get(f"rel::{code}::{psid}")
    if not r:
        return
    rec = ann.setdefault(psid, {})
    rec["relevance"] = r
    rec["has_5tag"] = st.session_state.get(f"has5::{code}::{psid}") or ""
    sflag = st.session_state.get(f"sflag::{code}::{psid}", "")
    rec["flag"] = sflag
    rec["note"] = (st.session_state.get(f"snote::{code}::{psid}") or "").strip()
    ann[psid] = rec
    save_ann(split, code, ann)


done = sum(1 for s in sample["sentence_id"] if ann.get(s, {}).get("relevance"))
with st.sidebar:
    st.progress(done / len(sample), text=f"{done}/{len(sample)} kalimat dinilai")
    jump = st.number_input("Ke kalimat #", 1, len(sample), st.session_state.idx + 1)
    target = int(jump) - 1
    if target != st.session_state.idx:                # persist the sentence we leave
        persist_sid(sample.iloc[st.session_state.idx]["sentence_id"])
        st.session_state.pills_nonce += 1
        st.session_state.idx = target
        st.rerun()
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
nonce = st.session_state.pills_nonce
sel_key = f"pills::{sid}::{nonce}"
sel = sorted(st.session_state.get(sel_key, []) or [])
sel_set = set(sel)
committed = committed_token_idxs(toks, cur.get("spans", []))

with st.expander("📖 Pedoman ringkas (lima tag, urutan keputusan, batas span)"):
    for c, full in TAGS.items():
        st.markdown(f"**{c} · {full}** — {TAG_HELP[c]}")
    st.markdown("**Urutan keputusan:** 1) nama teknologi? → TT · 2) tindakan teknis? "
                "→ SH · 3) konsep yang dipahami? → KN · 4) komunikasi/kerja sama? → SS "
                "· 5) sifat/watak? → WB.")
    st.markdown("**Batas span:** ambil span terpendek yang tetap bermakna; jangan "
                "sertakan pembuka (must, able to, experience in); pisahkan tindakan "
                "dari teknologi; tanda baca di luar span. Pendidikan/pengalaman/"
                "sertifikasi/syarat bahasa → bukan tag (penanda OS, has_5tag=TIDAK).")

st.subheader(f"Kalimat {i + 1} / {len(sample)}")
st.caption(f"{rec['country_iso']} · {rec['site']} · bahasa {rec['posting_language']} "
           f"· {rec['quarter']} · {sid}")
ctx = []
if rec["previous_sentence"]:
    ctx.append(f"<span style='color:#999'>… {esc(rec['previous_sentence'])}</span>")
ctx.append(f"<div style='font-size:1.3rem;font-weight:600;line-height:1.6;margin:"
           f".4rem 0'>{highlight(rec['sentence_text'], toks, sel_set, committed)}</div>")
if rec["next_sentence"]:
    ctx.append(f"<span style='color:#999'>{esc(rec['next_sentence'])} …</span>")
st.markdown("".join(ctx), unsafe_allow_html=True)
st.divider()

# --- sentence-level judgement (keyed per annotator+sentence; blank until chosen) --
c1, c2 = st.columns(2)
rel = c1.radio(
    "Relevansi kalimat", RELEVANCE,
    index=RELEVANCE.index(cur["relevance"]) if cur.get("relevance") in RELEVANCE else None,
    key=f"rel::{code}::{sid}")
has5 = c2.radio(
    "has_5tag (minimal satu dari lima tag?)", HAS_5TAG,
    index=HAS_5TAG.index(cur["has_5tag"]) if cur.get("has_5tag") in HAS_5TAG else None,
    horizontal=True, key=f"has5::{code}::{sid}")
sc1, sc2 = st.columns([1, 3])
sc1.selectbox("Penanda kalimat (opsional)", list(FLAGS),
              index=list(FLAGS).index(cur.get("flag", "")) if cur.get("flag", "") in FLAGS else 0,
              format_func=lambda c: FLAGS[c], key=f"sflag::{code}::{sid}")
sc2.text_input("Catatan kalimat (mis. OUT_OF_SCHEMA: pendidikan/pengalaman/sertifikasi)",
               value=cur.get("note", ""), key=f"snote::{code}::{sid}")
# Consistency prompts: the app no longer rewrites has_5tag, so any contradiction
# is shown to the annotator to resolve themselves (their answer is the datum).
if has5 is None:
    st.warning("has_5tag belum dijawab. Jawaban Anda sendiri yang dicatat — "
               "kolom ini tidak lagi diisi otomatis oleh sistem.")
elif has5 == "YA" and not cur.get("spans"):
    st.caption("has_5tag=YA berarti kalimat memuat ≥1 tag — tandai span-nya di bawah.")
elif has5 == "TIDAK" and cur.get("spans"):
    st.warning("Anda menjawab has_5tag=TIDAK tetapi sudah menandai span pada "
               "kalimat ini. Perbaiki salah satunya agar konsisten.")
if rel == "NON_RELEVANT" and has5 == "YA":
    st.warning("Pedoman Bagian 3.2: kalimat NON_RELEVANT seharusnya has_5tag=TIDAK.")

# --- span capture (only for RELEVANT sentences) ---------------------------------
st.markdown("#### Tandai span")
st.caption("🟨 kuning = sedang dipilih · 🟩 hijau = span yang sudah disimpan")
if rel != "RELEVANT":
    st.info("Span hanya untuk kalimat RELEVANT. Pilih RELEVANT di atas untuk menandai span.")
# st.pills identifies chips by their displayed label, so repeated words collide;
# append i zero-width spaces to make every label unique while looking identical.
st.pills("Klik kata-kata entitas (berdampingan)", options=list(range(len(toks))),
         format_func=lambda n: toks[n][0] + "​" * n,
         selection_mode="multi", key=sel_key,
         disabled=(rel != "RELEVANT"))
span_text, span_start, span_end = "", 0, 0
if sel:
    if sel != list(range(sel[0], sel[-1] + 1)):
        st.warning("Kata tidak berdampingan — span diambil dari kata pertama s.d. terakhir.")
    span_start, span_end = trim_span(
        rec["sentence_text"], toks[sel[0]][1], toks[sel[-1]][2])
    span_text = rec["sentence_text"][span_start:span_end]
    if span_start == span_end:
        st.warning("Pilihan hanya berisi tanda baca — tidak ada teks span.")
    else:
        st.success(f"Span terpilih: «{span_text}»  ({span_start}:{span_end})")

s1, s2, s3 = st.columns(3)
label = s1.selectbox("Label", [""] + list(TAGS), key=f"label::{nonce}",
                     format_func=lambda c: "— pilih label —" if c == "" else f"{c} — {TAGS[c]}")
conf = s2.selectbox("Keyakinan", [""] + CONFIDENCE, key=f"conf::{nonce}",
                    format_func=lambda x: "— pilih keyakinan —" if x == "" else f"{x} — {CONF_WORD[x]}")
flag = s3.selectbox("Penanda span", list(FLAGS), key=f"flag::{nonce}",
                    format_func=lambda c: FLAGS[c])
parent = ""
if flag == "OP" and cur.get("spans"):
    parent = st.selectbox("Parent span (tumpang tindih / OVERLAP)",
                          [""] + list(range(1, len(cur["spans"]) + 1)),
                          key=f"parent::{nonce}")
note = st.text_input("Catatan span (wajib bila ada penanda / keyakinan ≤ 2)",
                     key=f"note::{nonce}")

note_required = flag != "" or (conf != "" and conf <= 2)
add_disabled = (rel != "RELEVANT" or not sel or span_start == span_end
                or label == "" or conf == "" or (note_required and not note.strip()))
if st.button("➕ Tambah span", disabled=add_disabled, type="primary"):
    cur.setdefault("spans", []).append(
        {"span_text": span_text, "start_char": span_start, "end_char": span_end,
         "label": label, "confidence": conf, "flag": flag,
         "parent_span_id": parent, "note": note.strip()})
    cur["relevance"] = "RELEVANT"
    # has_5tag is NOT inferred from the presence of a span — it stays the
    # annotator's own answer so the agreement statistic measures their judgement
    # rather than this UI's bookkeeping. A contradiction (span present but
    # has_5tag=TIDAK) is surfaced below for the annotator to resolve themselves.
    cur["has_5tag"] = st.session_state.get(f"has5::{code}::{sid}") or ""
    cur["flag"] = st.session_state.get(f"sflag::{code}::{sid}", "")
    cur["note"] = (st.session_state.get(f"snote::{code}::{sid}") or "").strip()
    ann[sid] = cur
    save_ann(split, code, ann)
    st.session_state.pills_nonce += 1        # fresh chips + span widgets
    st.rerun()
if rel == "RELEVANT" and sel and label == "":
    st.caption("Pilih label dulu untuk menambah span.")
if rel == "RELEVANT" and sel and conf == "":
    st.caption("Pilih tingkat keyakinan dulu.")
if note_required and not note.strip():
    st.caption("Catatan wajib diisi karena ada penanda atau keyakinan ≤ 2.")

if cur.get("spans"):
    st.markdown("**Span pada kalimat ini:**")
for n, s in enumerate(cur.get("spans", [])):
    cols = st.columns([5, 3, 2, 1])
    cols[0].markdown(f"**{n + 1}.** «{esc(s['span_text'])}»")
    extra = (f" · {s['flag']}" if s.get("flag") else "") + \
            (f" · parent {s['parent_span_id']}" if s.get("parent_span_id") else "")
    cols[1].write(f"{s['label']} · conf {s['confidence']}{extra}")
    cols[2].caption(f"[{s['start_char']}:{s['end_char']}]")
    if cols[3].button("🗑", key=f"del{sid}{n}"):
        cur["spans"].pop(n)
        ann[sid] = cur
        save_ann(split, code, ann)
        st.rerun()

st.divider()
n1, n2, n3 = st.columns(3)
if n1.button("⟵ Sebelumnya", disabled=i == 0, use_container_width=True):
    persist_sid(sid)
    st.session_state.pills_nonce += 1
    st.session_state.idx = max(0, i - 1)
    st.rerun()
if n2.button("💾 Simpan kalimat", use_container_width=True):
    persist_sid(sid)
    st.success("Tersimpan." if st.session_state.get(f"rel::{code}::{sid}")
               else "Pilih relevansi dulu sebelum menyimpan.")
if n3.button("Berikutnya ⟶", disabled=i >= len(sample) - 1, use_container_width=True):
    persist_sid(sid)
    st.session_state.pills_nonce += 1
    st.session_state.idx = min(len(sample) - 1, i + 1)
    st.rerun()
