"""Public dashboard: a school reads a published competency profile (Gambar 3.3).

    .venv-ml\\Scripts\\streamlit.exe run dashboard_v3/public.py

It shows one versioned profile at a time: the recommendations with their
status, the evidence behind each one, the proposed KKNI level and okupasi, the
proposed coverage by the official references, the change log, and — when a
curriculum document is uploaded — its coverage and the delta recommendations.

It only READS profiles. It never runs the pipeline, never calls a paid model
and never collects postings. A draft profile is shown with a warning on every
page: it comes from an engineering run and is not a finding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_v3 import common as C                                       # noqa: E402

st.set_page_config(page_title="Profil Kompetensi RPL", layout="wide")

profiles = C.list_profiles()
if not profiles:
    st.title("Profil Kompetensi RPL")
    st.info("Belum ada profil yang diterbitkan. Operator menerbitkannya dari dasbor admin.")
    st.stop()

with st.sidebar:
    st.header("Profil")
    chosen = st.selectbox("Versi", profiles, format_func=lambda p: p["label"])
    profile = C.load_json(chosen["path"])
    recs = profile["recommendations"]
    st.header("Saring")
    status = st.multiselect("Status kelayakan", C.STATUS_ORDER, default=C.STATUS_ORDER,
                            format_func=lambda s: C.STATUS_LABEL[s])
    stream = st.multiselect("Jenis", ["technical", "soft"], default=["technical", "soft"],
                            format_func=lambda s: "Teknis" if s == "technical" else "Penunjang (soft skill)")
    level_options = [c for c in C.LEVEL_CODES + [C.NO_LEVEL]
                     if any(C.level_code(r) == c for r in recs)]
    levels = st.multiselect(
        "Tingkat kognitif (Bloom)", level_options, default=level_options,
        help="Tingkat yang diminta lowongan, dibaca dari kata kerja yang mereka pakai. "
             "Pilih tingkat yang sesuai dengan jenjang lembaga Anda.") if level_options else []
    min_postings = st.number_input("Minimal jumlah lowongan", min_value=0, value=0, step=1)
    query = st.text_input("Cari dalam pernyataan")

ident, corpus = profile["identity"], profile["corpus"]
st.title(ident["title"])
if ident.get("notice"):
    st.warning(ident["notice"] + " Angka di halaman ini belum boleh dikutip sebagai temuan.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lowongan dianalisis", f"{corpus['denominators']['n_postings']:,}")
c2.metric("Unit bukti", f"{corpus['n_evidence']:,}")
c3.metric("Kelompok bukti", f"{corpus['grouping']['n_groups']:,}")
c4.metric("Rekomendasi", f"{len(recs):,}")

shown = [r for r in recs
         if r["contract"]["status"] in status and r["stream"] in stream
         and r["n_postings"] >= min_postings
         and (C.level_code(r) is None or C.level_code(r) in levels)
         and (not query or query.lower() in r["statement"]["text"].lower())]

tab_list, tab_map, tab_summary, tab_curriculum, tab_changes, tab_about = st.tabs(
    ["Rekomendasi", "Peta pengetahuan & teknologi", "Ringkasan", "Kurikulum Anda",
     "Catatan perubahan", "Tentang"])

with tab_map:
    map_path = Path(chosen["path"]).parent / "term_map.json"
    if map_path.is_file():
        from dashboard_v3 import termmap_view
        termmap_view.render(st, C.load_json(map_path), recs)
    else:
        st.info("Profil ini diterbitkan sebelum peta kelompok tersedia. Terbitkan versi baru "
                "dari run yang memuat `term_map.json` untuk melihatnya.")

with tab_list:
    st.caption(f"{len(shown)} dari {len(recs)} rekomendasi. Urutan: status, lalu bobot "
               f"persistensi, lalu jumlah lowongan. Bobot hanya mengurutkan; tidak menaikkan status.")
    table = [{"#": r["rank"], "Pernyataan kompetensi": r["statement"]["text"],
              "Status": C.STATUS_LABEL[r["contract"]["status"]],
              "Lowongan": r["n_postings"], "Pangsa": f"{r['share_of_postings']:.1%}",
              "Tingkat kognitif": C.level_text(r.get("cognitive")),
              "KKNI (usulan)": (f"{r['kkni']['level']}" + ("" if r["kkni"].get("confident") else " ?"))
              if r.get("kkni") and r["kkni"].get("level") else "",
              "Rujukan resmi (usulan)": C.COVERAGE_LABEL.get(
                  ((r.get("coverage") or {}).get("SKKNI") or {}).get("proposed", ""), "")}
             for r in shown]
    st.dataframe(table, width="stretch", hide_index=True)
    st.subheader("Bukti di balik satu rekomendasi")
    if shown:
        pick = st.selectbox("Pilih pernyataan", shown,
                            format_func=lambda r: f"{r['rank']}. {r['statement']['text'][:110]}")
        a, b = st.columns(2)
        with a:
            st.markdown(f"**{pick['statement']['text']}**")
            st.write(f"Status: **{C.STATUS_LABEL[pick['contract']['status']]}**")
            if pick["contract"]["blockers"]:
                st.write("Belum naik status karena: " + ", ".join(pick["contract"]["blockers"]))
            st.write(f"Diminta oleh {pick['n_postings']} lowongan "
                     f"({pick.get('n_posting_groups', pick['n_postings'])} pemberi kerja/templat), "
                     f"{pick['share_of_postings']:.1%} dari lowongan yang dianalisis.")
            cog = pick.get("cognitive")
            default_verb = pick["statement"].get("verb_source") == "default"
            fam = pick.get("family")
            if fam:                              # one competency per level of a family
                n_fam = fam["n_postings"]
                if cog.get("level"):
                    st.write(f"Tingkat kognitif: **{C.level_text(cog)}** — kata kerja pada "
                             f"tingkat ini dipakai {cog['n_postings']} lowongan, "
                             f"{cog['share_of_family_postings']:.0%} dari {n_fam} lowongan "
                             f"rumpun kompetensi ini.")
                else:
                    st.write(f"Tingkat kognitif: **tidak dinyatakan lowongan** — "
                             f"{cog['n_postings']} lowongan menyebut pekerjaan ini tanpa kata "
                             f"kerja, sehingga tingkatnya tidak dapat dibaca.")
                if len(fam.get("competencies") or []) > 1:
                    st.markdown("**Kompetensi serumpun, per tingkat kognitif**")
                    st.dataframe([{"Tingkat": C.level_text(x) or "tidak dinyatakan",
                                   "Pernyataan kompetensi": x["statement"],
                                   "Lowongan": x["n_postings"],
                                   "": "← yang sedang dibuka"
                                   if x["competency_id"] == pick["competency_id"] else ""}
                                  for x in fam["competencies"]], hide_index=True, width="stretch")
                    st.caption("Satu lowongan dapat meminta lebih dari satu tingkat, sehingga "
                               "jumlahnya dapat melebihi jumlah lowongan rumpun.")
                if fam.get("description"):
                    with st.expander("Uraian rumpun (ditulis model bahasa, mencakup semua tingkat)"):
                        st.write(fam["description"]["text"])
                        st.caption("Uraian ini ditulis untuk seluruh rumpun dan mencampur "
                                   "beberapa tingkat; bukan pernyataan kompetensi satu tingkat.")
            elif cog is None:                    # a profile published before 2026-11-20
                st.write(f"Kata kerja: *{pick['statement']['verb']}* · tingkat kognitif "
                         f"(Krathwohl): {pick['statement']['krathwohl']}")
            elif cog.get("level"):
                st.write(f"Tingkat kognitif yang diminta lowongan: **{C.level_text(cog)}** — "
                         f"{cog['share']:.0%} dari {cog['n_postings_with_a_verb']} lowongan "
                         f"yang memakai kata kerja"
                         + (f". Tertinggi yang diminta sedikitnya seperlima lowongan: "
                            f"{C.level_text(cog['highest'])}."
                            if cog["highest"]["level"] != cog["level"] else "."))
                st.dataframe([{"Tingkat": C.level_text(lv), "Lowongan": lv["n_postings"],
                               "Pangsa": f"{lv['share']:.0%}"} for lv in cog["levels"]],
                             hide_index=True, width="stretch")
            else:
                st.write("Tingkat kognitif: **tidak dapat dibaca** — lowongan kelompok ini "
                         "hampir tidak memakai kata kerja.")
            st.write(f"Kata kerja pada pernyataan: *{pick['statement']['verb']}*"
                     + (" (bawaan metode, bukan dari lowongan; tidak menentukan tingkat)"
                        if default_verb else "")
                     + f" · keterpautan bukti {pick['statement']['grounding']:.2f}")
            kk = pick.get("kkni") or {}
            if kk.get("level"):
                st.write(f"Jenjang KKNI (usulan): **{kk['level']}**"
                         + ("" if kk.get("confident") else " — keyakinan rendah, menunggu telaah ahli"))
            if pick.get("okupasi"):
                st.write("Okupasi terdekat: " + "; ".join(
                    o.get("occupation_en") or "" for o in pick["okupasi"][:3]))
            cov = (pick.get("coverage") or {}).get("SKKNI")
            if cov:
                st.write(f"Rujukan resmi (usulan, belum divalidasi ahli): "
                         f"**{C.COVERAGE_LABEL.get(cov['proposed'], cov['proposed'])}**")
                st.dataframe([{"Unit rujukan": m["title"], "Kemiripan": m["similarity"]}
                              for m in cov["matches"]], hide_index=True, width="stretch")
        with b:
            st.markdown("**Frasa dari lowongan**")
            st.dataframe([{"Frasa": m["text"], "Tag": m["tag"], "Lowongan": m["n_postings"]}
                          for m in pick["evidence"]["members"][:15]], hide_index=True,
                         width="stretch")
            verbs = pick["evidence"].get("verbs") or []
            objects = pick["evidence"].get("knowledge_objects") or []
            if verbs or objects:
                st.markdown("**Kompetensi = kata kerja + pengetahuan**")
                if verbs:
                    st.write("Kata kerja yang dipakai lowongan: " + ", ".join(
                        f"{v['verb']} ({v['krathwohl']}, {v['n_postings']})" for v in verbs[:6]))
                if objects:
                    st.dataframe([{"Pengetahuan yang dikerjakan": o["text"],
                                   "Lowongan": o["n_postings"], "Pangsa": f"{o['share']:.0%}",
                                   "Kata kerja": ", ".join(o["verbs"])} for o in objects],
                                 hide_index=True, width="stretch")
            if pick["evidence"]["technologies"]:
                st.markdown("**Alat dan teknologi yang paling banyak dipakai**")
                st.write(", ".join(f"{t['text']} ({t['n_postings']})"
                                   for t in pick["evidence"]["technologies"]))
            for key, title in (("knowledge", "Pengetahuan lain yang diminta bersama"),
                               ("soft_skills", "Soft skill yang diharapkan"),
                               ("work_behaviours", "Perilaku kerja yang diharapkan")):
                rows = pick["evidence"].get(key) or []
                if rows:
                    st.markdown(f"**{title}**")
                    st.dataframe([{"": t["text"], "Lowongan": t["n_postings"],
                                   "Pangsa": f"{t['share']:.0%}",
                                   "Dibanding umum": f"{t['lift']}×" if t.get("lift") else ""}
                                  for t in rows], hide_index=True, width="stretch")
            if pick["evidence"].get("samples"):
                st.markdown("**Contoh kalimat sumber**")
                for s in pick["evidence"]["samples"][:3]:
                    st.caption(s["sentence_text"])
            t = pick.get("temporal")
            if t and t.get("shares"):
                st.markdown("**Pangsa per jendela waktu**")
                st.line_chart(t["shares"], height=160, color=C.BLUE)
                st.caption(f"Arah: {t['direction']} · volatilitas: {t['volatility']} · uji tren: {t['trend']}")

with tab_summary:
    a, b = st.columns(2)
    by_status = C.counts(recs, lambda r: C.STATUS_LABEL[r["contract"]["status"]])
    a.subheader("Status kelayakan")
    a.bar_chart(by_status, horizontal=True, color=C.BLUE)
    by_kkni = C.counts(recs, lambda r: f"KKNI {r['kkni']['level']}" if r.get("kkni") and r["kkni"].get("level") else None)
    if by_kkni:
        b.subheader("Jenjang KKNI (usulan)")
        b.bar_chart(dict(sorted(by_kkni.items())), horizontal=True, color=C.BLUE)
    by_cov = C.counts(recs, lambda r: C.COVERAGE_LABEL.get(
        ((r.get("coverage") or {}).get("SKKNI") or {}).get("proposed", ""), None))
    if by_cov:
        st.subheader("Cakupan oleh rujukan resmi (usulan, belum divalidasi ahli)")
        st.bar_chart(by_cov, horizontal=True, color=C.BLUE)

with tab_curriculum:
    st.write("Unggah dokumen kurikulum (teks atau PDF) untuk melihat kompetensi mana yang sudah "
             "tercakup dan usulan tindakan terhadap dokumen itu. Dokumen diproses di komputer "
             "ini dan tidak disimpan.")
    up = st.file_uploader("Dokumen kurikulum", type=["txt", "md", "pdf"])
    level = st.selectbox("Jenjang KKNI dokumen kurikulum (bila diketahui)",
                         [None, 2, 3, 4, 5, 6], format_func=lambda v: "tidak diisi" if v is None else str(v))
    if up is not None and st.button("Analisis cakupan"):
        from competency import coverage as CV
        from competency import delta as DL
        data = up.getvalue()
        units = (CV.units_from_curriculum_pdf(data) if up.name.lower().endswith(".pdf")
                 else CV.units_from_curriculum_text(data.decode("utf-8", errors="replace")))
        if not units:
            st.error("Tidak ada bagian yang dapat dibaca dari dokumen ini.")
        else:
            work = [dict(r, competency_id=r["competency_id"]) for r in recs]
            with st.spinner("Membandingkan dengan dokumen…"):
                CV.audit(work, units, "kurikulum", official=False)
                summary = DL.apply(work, "kurikulum", level)
            st.success(f"{len(units)} bagian dokumen dibandingkan dengan {len(work)} rekomendasi.")
            st.bar_chart({C.DELTA_LABEL[k]: v for k, v in summary["counts"].items() if v},
                         horizontal=True, color=C.BLUE)
            st.dataframe([{"Usulan": C.DELTA_LABEL[r["delta"]["type"]],
                           "Pernyataan kompetensi": r["statement"]["text"],
                           "Status": C.STATUS_LABEL[r["delta"]["effective_status"]],
                           "Dasar": " ".join(r["delta"]["basis"])}
                          for r in DL.priority_order(work)], hide_index=True,
                         width="stretch")
            st.caption("Semua ini usulan. Cakupan belum divalidasi ahli, dan usulan 'mengurangi "
                       "penekanan' tidak pernah melebihi status eksploratif tanpa telaah ahli.")

with tab_changes:
    log = profile["changelog"]
    if log.get("previous_version") is None:
        st.info("Versi pertama: belum ada catatan perubahan.")
    else:
        st.write(f"Dibandingkan dengan versi **{log['previous_version']}**.")
        for name, title in (("naik", "Naik status"), ("turun", "Turun status"),
                            ("masuk", "Masuk daftar"), ("keluar", "Keluar dari daftar")):
            st.subheader(f"{title} ({len(log[name])})")
            if log[name]:
                st.dataframe([{"Pernyataan": r["statement"], "Alasan": r["reason"]}
                              for r in log[name]], hide_index=True, width="stretch")

with tab_about:
    st.write(f"Sumber bukti: {corpus['evidence_source']}")
    st.write(f"Dibuat {ident['created']} dari run `{ident['source_run']}`.")
    st.markdown(
        "- Setiap rekomendasi dapat ditelusuri ke kalimat lowongan yang menjadi buktinya.\n"
        "- Status kelayakan mengikuti kontrak rekomendasi: *eksploratif* hanya untuk pemantauan; "
        "*kandidat tinjauan* layak ditelaah ahli; *siap ditelaah formal* memerlukan validasi ahli "
        "dan uji stabilitas, dan tidak pernah diberikan otomatis.\n"
        "- Jenjang KKNI, okupasi dan cakupan rujukan resmi adalah **usulan** sistem yang menunggu "
        "validasi ahli.\n"
        "- Profil ini bahan telaah, bukan keputusan kurikulum.")
