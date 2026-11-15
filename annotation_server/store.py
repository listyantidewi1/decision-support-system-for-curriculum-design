"""Durable SQLite store for gold-standard annotations.

Why per-sentence rows rather than one JSON blob per annotator:
  * two browser tabs (or a refresh mid-save) cannot clobber a whole file;
  * every write is an UPSERT on (split, annotator, sentence_id), so a retry is
    idempotent;
  * the append-only `events` table gives the dissertation a provenance trail —
    when each judgement was made, by whom, under which split — which is what
    lets independence (guideline Bagian 10.1) be demonstrated rather than
    asserted.

The SQLite file IS the deliverable: it is what the researcher downloads.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    split        TEXT NOT NULL,
    annotator    TEXT NOT NULL,
    sentence_id  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    updated_utc  TEXT NOT NULL,
    PRIMARY KEY (split, annotator, sentence_id)
);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT NOT NULL,
    split        TEXT NOT NULL,
    annotator    TEXT NOT NULL,
    sentence_id  TEXT,
    action       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ann_scope ON annotations(split, annotator);
CREATE INDEX IF NOT EXISTS idx_ev_scope  ON events(split, annotator);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def _conn(self):
        """Commit-and-CLOSE. `with sqlite3.connect(...)` only manages the
        transaction and leaves the handle open, which leaks a file descriptor per
        request on a long-running server (and blocks temp-dir cleanup on Windows).
        """
        c = sqlite3.connect(self.path, timeout=15)
        c.execute("PRAGMA journal_mode=WAL")      # concurrent readers + 1 writer
        c.execute("PRAGMA busy_timeout=15000")
        c.row_factory = sqlite3.Row
        try:
            with closing(c):
                with c:                            # commit on success, rollback on error
                    yield c
        finally:
            pass

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---------------------------------------------------------------- writes
    def put(self, split: str, annotator: str, sentence_id: str, payload: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO annotations (split, annotator, sentence_id, payload, "
                "updated_utc) VALUES (?,?,?,?,?) ON CONFLICT(split, annotator, "
                "sentence_id) DO UPDATE SET payload=excluded.payload, "
                "updated_utc=excluded.updated_utc",
                (split, annotator, sentence_id,
                 json.dumps(payload, ensure_ascii=False), _now()))
            c.execute("INSERT INTO events (ts_utc, split, annotator, sentence_id, "
                      "action) VALUES (?,?,?,?,?)",
                      (_now(), split, annotator, sentence_id, "put"))

    def log(self, split: str, annotator: str, action: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO events (ts_utc, split, annotator, sentence_id, "
                      "action) VALUES (?,?,?,?,?)",
                      (_now(), split, annotator, None, action))

    # ----------------------------------------------------------------- reads
    def get_all(self, split: str, annotator: str) -> dict:
        """Everything one annotator has saved, in the app's autosave shape."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT sentence_id, payload FROM annotations WHERE split=? AND "
                "annotator=?", (split, annotator)).fetchall()
        return {r["sentence_id"]: json.loads(r["payload"]) for r in rows}

    def progress(self, split: str | None = None) -> list[dict]:
        q = ("SELECT split, annotator, COUNT(*) AS n_sentences, "
             "MIN(updated_utc) AS first_saved, MAX(updated_utc) AS last_saved "
             "FROM annotations {} GROUP BY split, annotator ORDER BY split, annotator")
        where, args = ("WHERE split=?", (split,)) if split else ("", ())
        with self._conn() as c:
            return [dict(r) for r in c.execute(q.format(where), args).fetchall()]

    def splits(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT split FROM annotations ORDER BY split")]

    def export(self, split: str | None = None) -> dict:
        """{split: {annotator: {sentence_id: payload}}} — the download payload."""
        where, args = ("WHERE split=?", (split,)) if split else ("", ())
        with self._conn() as c:
            rows = c.execute(
                f"SELECT split, annotator, sentence_id, payload FROM annotations "
                f"{where}", args).fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["split"], {}).setdefault(r["annotator"], {})[
                r["sentence_id"]] = json.loads(r["payload"])
        return out

    def events(self, split: str | None = None, limit: int = 5000) -> list[dict]:
        where, args = ("WHERE split=?", (split,)) if split else ("", ())
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?",
                (*args, limit)).fetchall()]
