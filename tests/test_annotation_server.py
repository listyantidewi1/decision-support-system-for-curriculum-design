"""Tests for the annotation collector (store + token auth + API).

Run with the web venv, which has fastapi/httpx:
  .venv\\Scripts\\python.exe tests\\test_annotation_server.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_server.auth import mint, resolve  # noqa: E402
from annotation_server.store import Store  # noqa: E402

AV_TOK, AI_TOK, ADMIN_TOK = "tok-av-xxxxxxxx", "tok-ai-xxxxxxxx", "tok-admin-xxxxxxxx"
TOKENS = {
    AV_TOK: {"annotator": "AV", "split": "pilot_100"},
    AI_TOK: {"annotator": "AI", "split": "pilot_100"},
    "tok-av-practice": {"annotator": "AV", "split": "practice_10"},
    ADMIN_TOK: {"annotator": "ADMIN", "split": "pilot_100", "admin": True},
}


def _client(tmp: Path):
    os.environ["GS_TOKENS"] = json.dumps(TOKENS)
    os.environ["GS_DB_PATH"] = str(tmp / "ann.sqlite")
    import annotation_server.api as api
    api._store = None                      # fresh store per test db
    from fastapi.testclient import TestClient
    return TestClient(api.app)


# ------------------------------------------------------------------- store
def test_store_upsert_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "a.sqlite")
        rec = {"relevance": "RELEVANT", "has_5tag": "YA", "spans": []}
        s.put("pilot_100", "AV", "s1", rec)
        s.put("pilot_100", "AV", "s1", rec)          # retry must not duplicate
        assert s.get_all("pilot_100", "AV") == {"s1": rec}
        assert s.progress("pilot_100")[0]["n_sentences"] == 1


def test_store_scopes_by_split_and_annotator():
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "a.sqlite")
        s.put("pilot_100", "AV", "s1", {"relevance": "RELEVANT"})
        s.put("practice_10", "AV", "s1", {"relevance": "NON_RELEVANT"})
        s.put("pilot_100", "AI", "s1", {"relevance": "UNCERTAIN"})
        # practice work must never leak into the pilot split
        assert s.get_all("pilot_100", "AV")["s1"]["relevance"] == "RELEVANT"
        assert s.get_all("practice_10", "AV")["s1"]["relevance"] == "NON_RELEVANT"
        assert s.get_all("pilot_100", "AI")["s1"]["relevance"] == "UNCERTAIN"
        assert set(s.splits()) == {"pilot_100", "practice_10"}


def test_store_records_an_audit_trail():
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "a.sqlite")
        s.put("pilot_100", "AV", "s1", {"relevance": "RELEVANT"})
        ev = s.events("pilot_100")
        assert ev and ev[0]["action"] == "put" and ev[0]["annotator"] == "AV"
        assert ev[0]["ts_utc"]


# -------------------------------------------------------------------- auth
def test_token_resolution_and_unknown_tokens():
    os.environ["GS_TOKENS"] = json.dumps(TOKENS)
    p = resolve(AV_TOK)
    assert p and p.annotator == "AV" and p.split == "pilot_100" and not p.admin
    assert resolve(ADMIN_TOK).admin is True
    assert resolve("nope") is None
    assert resolve(None) is None
    assert resolve("") is None


def test_token_prefix_is_not_accepted():
    os.environ["GS_TOKENS"] = json.dumps(TOKENS)
    assert resolve(AV_TOK[:-1]) is None       # no prefix / partial match
    assert resolve(AV_TOK + "x") is None


def test_mint_produces_distinct_opaque_tokens():
    a, _ = mint("AV", "pilot_100")
    b, _ = mint("AV", "pilot_100")
    assert a != b and len(a) > 20
    _, entry = mint("ADMIN", "pilot_100", admin=True)
    assert entry["admin"] is True


# --------------------------------------------------------------------- api
def test_api_requires_a_token():
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        assert c.get("/health").json() == {"ok": True}      # health is open
        assert c.get("/annotations").status_code == 401
        assert c.get("/annotations", params={"token": "nope"}).status_code == 401


def test_api_identity_comes_from_the_token_not_the_client():
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        me = c.get("/me", params={"token": AV_TOK}).json()
        assert me["annotator"] == "AV" and me["split"] == "pilot_100"


def test_api_annotator_cannot_see_another_annotators_work():
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        c.put("/annotations/s1", params={"token": AV_TOK},
              json={"relevance": "RELEVANT", "has_5tag": "YA", "spans": []})
        assert "s1" in c.get("/annotations", params={"token": AV_TOK}).json()
        # AI scope is empty - no endpoint exposes AV work to AI
        assert c.get("/annotations", params={"token": AI_TOK}).json() == {}


def test_api_practice_token_cannot_write_into_the_pilot_split():
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        c.put("/annotations/s1", params={"token": "tok-av-practice"},
              json={"relevance": "RELEVANT"})
        # the same annotator PILOT scope stays clean
        assert c.get("/annotations", params={"token": AV_TOK}).json() == {}


def test_api_rejects_non_object_bodies():
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        r = c.put("/annotations/s1", params={"token": AV_TOK}, json=[1, 2, 3])
        assert r.status_code == 400


def test_admin_endpoints_need_admin_token():
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        assert c.get("/admin/progress", params={"token": AV_TOK}).status_code == 403
        assert c.get("/admin/export", params={"token": AV_TOK}).status_code == 403
        assert c.get("/admin/progress", params={"token": ADMIN_TOK}).status_code == 200


def test_admin_export_zip_is_iaa_ready():
    import io
    import zipfile
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        c.put("/annotations/s1", params={"token": AV_TOK},
              json={"relevance": "RELEVANT", "has_5tag": "YA", "spans": []})
        c.put("/annotations/s1", params={"token": AI_TOK},
              json={"relevance": "RELEVANT", "has_5tag": "YA", "spans": []})
        r = c.get("/admin/export", params={"token": ADMIN_TOK, "split": "pilot_100"})
        assert r.status_code == 200
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = set(z.namelist())
        # names must match what gold_standard.iaa --ann-dir expects
        assert {"pilot_100_AV.json", "pilot_100_AI.json"} <= names
        assert "events.csv" in names and "progress.json" in names
        assert json.loads(z.read("pilot_100_AV.json"))["s1"]["relevance"] == "RELEVANT"


def test_admin_progress_reports_per_annotator_counts():
    with tempfile.TemporaryDirectory() as d:
        c = _client(Path(d))
        for sid in ("s1", "s2", "s3"):
            c.put(f"/annotations/{sid}", params={"token": AV_TOK},
                  json={"relevance": "RELEVANT"})
        c.put("/annotations/s1", params={"token": AI_TOK}, json={"relevance": "RELEVANT"})
        prog = c.get("/admin/progress", params={"token": ADMIN_TOK}).json()["progress"]
        by = {p["annotator"]: p["n_sentences"] for p in prog}
        assert by["AV"] == 3 and by["AI"] == 1      # started-vs-empty is visible


def _run_standalone() -> int:
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed, failed = 0, []
    for name, fn in fns.items():
        try:
            fn()
            passed += 1
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
