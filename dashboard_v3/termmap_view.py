"""The cluster map of knowledge terms and of tools and technology, as a page.

Reads `term_map.json` (written by `competency.termmap`, copied next to a
published profile). It only draws: every number on the page is in that file.

Reading the picture: a point is one term; its size is the number of postings
that ask for it; points are close when the same postings ask for both. Circles
are tools and technology, squares are knowledge terms. The eight largest
clusters have a colour each; the others are grey. The table under the picture
says the same thing in words.
"""
from __future__ import annotations

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#9aa0a6"
LABEL_INK = "#6f7680"
KIND = {"TT": "Alat & teknologi", "KN": "Pengetahuan"}
OTHER = "Kelompok lain"


def cluster_names(tm: dict) -> dict[int, str]:
    return {c["cluster"]: f"{c['cluster']}. {c['label']}" for c in tm["clusters"]}


def frame(tm: dict):
    """One row per term, with the display columns. A cluster keeps its colour
    whatever the filters are: the colour belongs to the cluster, not to its rank
    among the points currently shown."""
    import pandas as pd
    names = cluster_names(tm)
    coloured = [c["cluster"] for c in tm["clusters"][:len(PALETTE)]]
    df = pd.DataFrame(tm["terms"])
    df["Jenis"] = df["tag"].map(KIND)
    df["Kelompok"] = df["cluster"].map(names)
    df["Warna"] = df["cluster"].map(lambda c: names[c] if c in coloured else OTHER)
    domain = [names[c] for c in coloured] + ([OTHER] if len(tm["clusters"]) > len(coloured) else [])
    colours = PALETTE[:len(coloured)] + ([GREY] if len(domain) > len(coloured) else [])
    return df, domain, colours


def related_competencies(recs: list[dict], terms: set[str], top: int = 15) -> list[dict]:
    """Competencies whose tools or knowledge include terms of the cluster."""
    low = {t.lower() for t in terms}
    rows = []
    for r in recs:
        ev = r.get("evidence", {})
        hit = sorted({t["text"] for key in ("technologies", "knowledge_objects", "knowledge")
                      for t in ev.get(key) or [] if t["text"].lower() in low})
        if hit:
            rows.append({"Pernyataan kompetensi": r["statement"]["text"],
                         "Lowongan": r["n_postings"], "Istilah kelompok ini": ", ".join(hit),
                         "_n": len(hit)})
    rows.sort(key=lambda x: (-x["_n"], -x["Lowongan"]))
    return [{k: v for k, v in x.items() if k != "_n"} for x in rows[:top]]


def render(st, tm: dict, recs: list[dict]) -> None:
    import altair as alt
    st.caption(
        "Setiap titik adalah satu istilah. Ukuran titik = jumlah lowongan yang memintanya. Dua "
        "titik berdekatan bila lowongan yang sama meminta keduanya. Lingkaran = alat & teknologi, "
        "persegi = pengetahuan. Kelompok dibentuk dari kemunculan bersama dalam lowongan, bukan "
        "dari kemiripan makna; letak titik hanya alat bantu gambar.")
    df, domain, colours = frame(tm)
    if df.empty:
        st.info("Belum ada istilah yang cukup sering muncul untuk dipetakan.")
        return
    names = cluster_names(tm)
    a, b, c = st.columns([2, 2, 3])
    kind = a.radio("Tampilkan", ["Semua", KIND["KN"], KIND["TT"]], horizontal=True,
                   key="termmap_kind")
    lo, hi = int(df["n_postings"].min()), int(df["n_postings"].max())
    floor = b.slider("Minimal jumlah lowongan", min_value=lo, max_value=max(hi, lo + 1), value=lo,
                     key="termmap_floor") if hi > lo else lo
    pick = c.selectbox("Sorot satu kelompok", [None] + [x["cluster"] for x in tm["clusters"]],
                       format_func=lambda v: "Semua kelompok" if v is None else names[v],
                       key="termmap_cluster")
    shown = df[df["n_postings"] >= floor]
    if kind != "Semua":
        shown = shown[shown["Jenis"] == kind]
    st.caption(f"{len(shown)} dari {len(df)} istilah · {tm['n_postings']:,} lowongan · "
               f"{len(tm['clusters'])} kelompok")
    if shown.empty:
        st.info("Tidak ada istilah dengan saringan ini.")
        return
    faded = (alt.condition(alt.datum.cluster == pick, alt.value(0.95), alt.value(0.10))
             if pick is not None else alt.value(0.85))
    base = alt.Chart(shown)
    points = base.mark_point(filled=True, stroke="white", strokeWidth=0.7).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        size=alt.Size("n_postings:Q", scale=alt.Scale(type="sqrt", range=[45, 1100]), legend=None),
        color=alt.Color("Warna:N", scale=alt.Scale(domain=domain, range=colours),
                        legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=460)),
        shape=alt.Shape("Jenis:N", scale=alt.Scale(domain=[KIND["TT"], KIND["KN"]],
                                                   range=["circle", "square"]),
                        legend=alt.Legend(title=None, orient="top")),
        opacity=faded,
        tooltip=[alt.Tooltip("term:N", title="Istilah"), alt.Tooltip("Jenis:N"),
                 alt.Tooltip("n_postings:Q", title="Lowongan", format=","),
                 alt.Tooltip("share:Q", title="Pangsa lowongan", format=".1%"),
                 alt.Tooltip("Kelompok:N")])
    # names on the picture: the most asked-for terms, or all of the cluster in focus
    named = (shown[shown["cluster"] == pick] if pick is not None
             else shown.nlargest(45, "n_postings"))
    text = alt.Chart(named).mark_text(dy=-11, fontSize=11, color=LABEL_INK).encode(
        x="x:Q", y="y:Q", text="term:N")
    st.altair_chart((points + text).properties(height=640).interactive(), width="stretch")

    if pick is None:
        st.subheader("Kelompok")
        st.dataframe([{"Kelompok": names[x["cluster"]], "Istilah": x["n_terms"],
                       KIND["KN"]: x["n_knowledge"], KIND["TT"]: x["n_technology"],
                       "Istilah terbanyak": ", ".join(x["top_terms"][:8])}
                      for x in tm["clusters"]], hide_index=True, width="stretch")
        return
    members = df[df["cluster"] == pick].sort_values("n_postings", ascending=False)
    st.subheader(names[pick])
    left, right = st.columns(2)
    for col, tag in ((left, "TT"), (right, "KN")):
        part = members[members["tag"] == tag]
        col.markdown(f"**{KIND[tag]}** ({len(part)})")
        col.dataframe([{"Istilah": r.term, "Lowongan": r.n_postings, "Pangsa": f"{r.share:.1%}"}
                       for r in part.itertuples()], hide_index=True, width="stretch")
    related = related_competencies(recs, set(members["term"]))
    if related:
        st.markdown("**Kompetensi yang memakai istilah kelompok ini**")
        st.dataframe(related, hide_index=True, width="stretch")
