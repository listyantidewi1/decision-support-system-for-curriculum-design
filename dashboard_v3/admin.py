"""Admin dashboard: the operator's view of runs, audits and human review queues.

    .venv-ml\\Scripts\\streamlit.exe run dashboard_v3/admin.py --server.port 8502

What an operator can do here:
  * look at every competency run: its counts, its figures, the validity status
    of each evidence group;
  * work through the two human review queues the pipeline produces — the D1/D2
    conflicts routed to a person, and the competency statements — and save the
    decisions next to the run;
  * publish a versioned profile from a run (as a draft unless the box "sah untuk
    dirilis" is ticked: that box is a person's decision about the study, never
    a default);
  * read the experiment log and see what the monthly collection would do.

What it deliberately cannot do: start a collection round, call a paid model, or
touch the gold standard. Those are run from the command line by a person who
has read what they do.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_v3 import common as C                                       # noqa: E402

st.set_page_config(page_title="Admin — Profil Kompetensi RPL", layout="wide")
st.title("Dasbor admin")

runs = C.list_runs()
page = st.sidebar.radio("Halaman", ["Run", "Audit kelompok", "Telaah konflik D1/D2",
                                    "Telaah pernyataan", "Terbitkan profil",
                                    "Log eksperimen", "Koleksi lowongan"])
run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name) if runs else None


def need_run():
    if run is None:
        st.info("Belum ada run. Jalankan `python -m competency.pipeline e0 --tag <nama>`.")
        st.stop()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if page == "Run":
    need_run()
    report = C.load_json(run / "report.json")
    st.warning(report["notice"])
    a, b, c, d = st.columns(4)
    a.metric("Lowongan", f"{report['denominators']['n_postings']:,}")
    b.metric("Unit bukti", f"{report['n_evidence']:,}")
    c.metric("Kelompok", f"{report['grouping']['n_groups']:,}")
    d.metric("Pernyataan", f"{report['n_competencies']:,}")
    st.write(f"Sumber bukti: {report['evidence_source']} · waktu proses {report['seconds']:.0f} s")
    st.json({k: report[k] for k in ("n_evidence_by_tag", "status_distribution", "contract",
                                    "mapping", "temporal", "coverage") if k in report},
            expanded=False)
    figs = sorted((run / "figures").glob("*.png")) if (run / "figures").is_dir() else []
    if figs:
        for f in figs:
            st.image(str(f))
    else:
        st.caption(f"Belum ada gambar. Buat dengan: python -m competency.figures run {run}")

elif page == "Audit kelompok":
    need_run()
    groups = C.load_json(run / "groups.json")
    st.bar_chart(C.counts(groups, lambda g: g["status"]), horizontal=True, color=C.BLUE)
    pick = st.multiselect("Status", sorted({g["status"] for g in groups}),
                          default=sorted({g["status"] for g in groups} - {"sahih"}))
    st.dataframe([{"Kelompok": g["group_id"], "Status": g["status"], "Label": g["label"],
                   "Lowongan": g["n_postings"], "Kohesi": round(g["cohesion"], 2),
                   "Catatan": ", ".join(g["status_flags"]),
                   "Anggota": "; ".join(m["text"] for m in g["members"][:6])}
                  for g in groups if g["status"] in pick], hide_index=True,
                 width="stretch")

elif page == "Telaah konflik D1/D2":
    need_run()
    sheet = run / "d3_conflicts_for_human_review.csv"
    if not sheet.is_file():
        st.info(f"Belum ada lembar konflik. Buat dengan: python -m fivetag.conflict_routing {run}")
        st.stop()
    rows = read_csv(sheet)
    final = [k for k in rows[0] if k.startswith("FINAL")][0]
    done = sum(1 for r in rows if r[final].strip())
    st.write(f"{len(rows)} konflik menunggu keputusan manusia; {done} sudah diisi. "
             f"Aturan risiko: `docs/experiments/PREREG_d3_conflict_routing.md`.")
    edited = st.data_editor(rows, hide_index=True, width="stretch",
                            disabled=[k for k in rows[0] if k not in (final, "note")],
                            column_config={final: st.column_config.SelectboxColumn(
                                options=["", "SH", "SS", "KN", "TT", "WB", "NONE"])})
    if st.button("Simpan keputusan"):
        write_csv(sheet, edited)
        st.success("Tersimpan.")

elif page == "Telaah pernyataan":
    need_run()
    comps = C.load_json(run / "competencies.json")
    path = run / "review_statements.csv"
    old = {r["competency_id"]: r for r in read_csv(path)} if path.is_file() else {}
    rows = [{"competency_id": c["competency_id"], "pernyataan": c["statement"]["text"],
             "status": c["contract"]["status"], "lowongan": c["n_postings"],
             "keputusan": old.get(c["competency_id"], {}).get("keputusan", ""),
             "perbaikan": old.get(c["competency_id"], {}).get("perbaikan", ""),
             "catatan": old.get(c["competency_id"], {}).get("catatan", "")} for c in comps]
    st.write("Keputusan penelaah disimpan di samping run dan tidak mengubah keluaran pipeline. "
             "Status 'siap ditelaah formal' tetap memerlukan telaah ahli yang tercatat.")
    edited = st.data_editor(rows, hide_index=True, width="stretch",
                            disabled=["competency_id", "pernyataan", "status", "lowongan"],
                            column_config={"keputusan": st.column_config.SelectboxColumn(
                                options=["", "terima", "perbaiki", "tolak"])})
    if st.button("Simpan telaah"):
        write_csv(path, edited)
        st.success(f"Tersimpan di {path.name}.")

elif page == "Terbitkan profil":
    need_run()
    profiles = C.list_profiles()
    st.write("Sebuah versi ditulis satu kali. Koreksi diterbitkan sebagai versi baru.")
    version = st.text_input("Versi (mis. 2026-2)")
    previous = st.selectbox("Bandingkan dengan versi sebelumnya", [None] + profiles,
                            format_func=lambda p: "tidak ada" if p is None else p["label"])
    valid = st.checkbox("Sah untuk dirilis (korpus sah, telaah ahli selesai, stabilitas teruji)",
                        value=False)
    if st.button("Terbitkan") and version.strip():
        from competency import profile as PF
        try:
            p = PF.build(run, version.strip(), previous=previous["path"] if previous else None,
                         valid_for_release=valid, profiles_dir=C.profiles_dir())
            st.success(f"Diterbitkan sebagai {p['identity']['kind']}: {p['out_dir']}")
        except PF.ProfileError as exc:
            st.error(str(exc))
    st.dataframe([{"Versi": p["version"], "Jenis": p["kind"]} for p in profiles], hide_index=True)

elif page == "Log eksperimen":
    path = C.REPO_ROOT / "docs" / "experiments" / "EXPERIMENT_LOG.md"
    st.markdown(path.read_text(encoding="utf-8") if path.is_file() else "Log tidak ditemukan.")
    figs = sorted((C.REPO_ROOT / "docs" / "experiments" / "figures").glob("*.png"))
    for f in figs:
        st.image(str(f))

elif page == "Koleksi lowongan":
    st.write("Hanya melihat. Koleksi dijalankan dari baris perintah: "
             "`python -m acquisition.monthly run --go`.")
    if st.button("Lihat status koleksi bulanan"):
        out = subprocess.run([sys.executable, "-m", "acquisition.monthly", "status"],
                             cwd=C.REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        st.code(out.stdout or out.stderr)
