"""Tests for governance.zoning: which sentences of a posting state a duty or a
requirement. Every sentence here is invented.

Run:  .venv-ml\\Scripts\\python.exe tests\\test_zoning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from governance import zoning as Z                                            # noqa: E402


def test_headers_are_recognised_in_english_and_indonesian_and_nothing_else_is():
    for text, zone in [("Responsibilities", "duty"), ("Key Responsibilities:", "duty"),
                       ("What you'll do", "duty"), ("What you’ll do", "duty"),
                       ("Deskripsi Pekerjaan", "duty"), ("Tanggung jawab utama", "duty"),
                       ("Requirements", "requirement"), ("Preferred Qualifications", "requirement"),
                       ("What we're looking for", "requirement"), ("Kualifikasi:", "requirement"),
                       ("Benefits", "benefit"), ("What we offer", "benefit"),
                       ("What's in it for you?", "benefit"), ("Fasilitas", "benefit"),
                       ("About us", "company"), ("Company Description", "company"),
                       ("Tentang kami", "company"), ("Additional Information", "admin"),
                       ("Job Type: Full-time", "admin"), ("How to apply", "admin"),
                       ("* Requirements *", "requirement")]:
        assert Z.header_zone(text) == zone, text
    for text in ["Develop REST APIs with Spring Boot.", "Strong knowledge of data structures.",
                 "We offer health insurance and a yearly bonus for every employee of the company.",
                 "Requirements gathering with stakeholders", "Benefits of unit testing", ""]:
        assert Z.header_zone(text) is None, text
    assert set(Z.KEEP_ZONES) | set(Z.DROP_ZONES) == set(Z._HEADERS)


def test_a_sentence_takes_the_zone_of_the_last_header_before_it_in_its_posting():
    rows = [("p1", 0, "Join our growing team."),                # before any header
            ("p1", 1, "Responsibilities"),
            ("p1", 2, "Develop REST APIs."),
            ("p1", 3, "Benefits"),
            ("p1", 4, "Support you and your family."),
            ("p2", 0, "Maintain databases."),                    # another posting: no header yet
            ("p2", 1, "About us"),
            ("p2", 2, "We build on our values of respect.")]
    shuffled = [rows[i] for i in (4, 0, 7, 2, 5, 1, 3, 6)]       # input order must not matter
    weak = Z.weak_labels([r[0] for r in shuffled], [r[1] for r in shuffled],
                         [r[2] for r in shuffled])
    got = {(r[0], r[1]): w for r, w in zip(shuffled, weak)}
    assert got == {("p1", 0): None, ("p1", 1): "header", ("p1", 2): "duty", ("p1", 3): "header",
                   ("p1", 4): "benefit", ("p2", 0): None, ("p2", 1): "header",
                   ("p2", 2): "company"}


def _corpus(n=60):
    """Postings with headers teach the classifier; postings without need it."""
    pids, idx, texts = [], [], []

    def add(p, lines):
        for i, t in enumerate(lines):
            pids.append(p)
            idx.append(i)
            texts.append(t)
    duties = ["Develop REST APIs for internal services", "Maintain relational databases",
              "Write unit tests and review code", "Deploy services with Docker and Kubernetes",
              "Debug production issues in Python code", "Design database schemas"]
    perks = ["Health insurance for you and your family", "Annual bonus and paid leave",
             "Free lunch and a gym membership", "Flexible hours and remote work allowance",
             "We believe in an inclusive workplace", "Our mission is to delight customers"]
    for p in range(n):
        add(f"h{p}", ["Responsibilities", duties[p % 6], duties[(p + 1) % 6], "Benefits",
                      perks[p % 6], perks[(p + 2) % 6]])
    add("x1", ["Develop REST APIs and maintain databases in Python"])
    add("x2", ["Health insurance, annual bonus and free lunch for your family"])
    return pids, idx, texts


def test_headers_decide_below_them_and_the_classifier_decides_where_there_is_none():
    pids, idx, texts = _corpus()
    res = Z.fit(pids, idx, texts, Z.ZoningConfig(min_df=1, drop_below=0.5))
    keep = dict(zip(zip(pids, idx), res["keep"]))
    assert keep[("h0", 1)] and keep[("h0", 2)]                  # under "Responsibilities"
    assert not keep[("h0", 4)] and not keep[("h0", 5)]          # under "Benefits"
    assert not keep[("h0", 0)] and not keep[("h0", 3)]          # the headers themselves
    assert keep[("x1", 0)] and not keep[("x2", 0)]              # no header: the classifier
    r = res["report"]
    assert r["n_headers"] == 120 and r["n_postings_with_a_header"] == 60 and r["n_postings"] == 62
    assert r["weak_zone_counts"]["duty"] == 120 and r["weak_zone_counts"]["benefit"] == 120
    assert r["decisions"]["classifier_keep"] == 1 and r["decisions"]["classifier_drop"] == 1
    held = r["agreement_with_weak_labels_on_held_out_postings"]
    assert held["accuracy"] >= 0.95 and "not accuracy against people" in held["note"]
    assert r["n_kept"] + r["n_dropped"] == len(texts)


def test_the_classifier_drops_only_when_it_is_confident():
    pids, idx, texts = _corpus()
    pids.append("x3")
    idx.append(0)
    texts.append("Quarterly planning with the wider group")      # resembles neither side
    loose = Z.fit(pids, idx, texts, Z.ZoningConfig(min_df=1, drop_below=0.5))
    strict = Z.fit(pids, idx, texts, Z.ZoningConfig(min_df=1, drop_below=0.02))
    assert strict["report"]["n_dropped"] <= loose["report"]["n_dropped"]
    assert strict["keep"][-1] is True                            # unsure means keep
    assert 0.0 <= loose["p_keep"][-1] <= 1.0


def test_without_headers_nothing_is_dropped_and_the_report_says_why():
    res = Z.fit(["a", "a", "b"], [0, 1, 0], ["Develop APIs.", "Free lunch.", "Maintain servers."])
    assert res["keep"] == [True, True, True]
    assert res["report"]["classifier"].startswith("none")
    assert res["report"]["decisions"]["kept_without_evidence"] == 3


def test_the_module_says_what_it_cannot_claim_and_stays_out_of_the_gold_standard():
    src = (PROJECT_ROOT / "governance" / "zoning.py").read_text(encoding="utf-8")
    assert "not validated against people" in src.lower() or "It is not validated" in src
    assert "to_parquet" not in src and "write_text" not in src   # it decides; it writes nothing
    assert Z.ZoningConfig().drop_below == 0.5 and Z.ZONING_VERSION == "zoning-v1"


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
