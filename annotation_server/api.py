"""FastAPI collector for gold-standard annotations.

One small service, one SQLite file. The Streamlit annotator talks to it instead
of writing to local disk, so the annotations survive a host restart (every free
Streamlit/Spaces host has an ephemeral filesystem) and the researcher downloads
them from one place instead of chasing three people for files.

Endpoints
  GET  /health                     liveness, no auth
  GET  /me                         who this token is (annotator + split)
  GET  /sentences                  the caller's own split's sentences to annotate
  GET  /annotations                caller's own saved work, for resume
  PUT  /annotations/{sentence_id}  upsert one sentence (idempotent)
  GET  /admin/progress             per-annotator counts + first/last save
  GET  /admin/export               ZIP: one JSON per annotator + events.csv
  GET  /admin/export.json          same payload as JSON

Auth is a bearer token per annotator (see auth.py). An annotator can only read
and write their OWN (annotator, split) scope — the API never lets one annotator
see another's work, which is what keeps the three judgements independent
(guideline Bagian 10.1) rather than merely asking them to be.
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import Principal, resolve
from .samples import SampleUnavailable, load_rows
from .store import Store

DEFAULT_DB_PATH = "data/annotations.sqlite"
app = FastAPI(title="Gold-standard annotation collector", version="1.0.0")
_store: Store | None = None
_store_path: str | None = None


def store() -> Store:
    """Resolve GS_DB_PATH at CALL time, not import time, and re-open when it
    changes. Freezing it at import made the path unreconfigurable after the first
    import, so every caller silently shared whichever database was configured
    first."""
    global _store, _store_path
    path = os.environ.get("GS_DB_PATH", DEFAULT_DB_PATH)
    if _store is None or _store_path != path:
        _store, _store_path = Store(path), path
    return _store


def _bearer(request: Request, token: str | None = Query(default=None)) -> str | None:
    hdr = request.headers.get("authorization", "")
    if hdr.lower().startswith("bearer "):
        return hdr[7:].strip()
    return token


def principal(request: Request, token: str | None = Query(default=None)) -> Principal:
    p = resolve(_bearer(request, token))
    if p is None:
        raise HTTPException(status_code=401, detail="unknown or missing token")
    return p


def admin(p: Principal = Depends(principal)) -> Principal:
    if not p.admin:
        raise HTTPException(status_code=403, detail="admin token required")
    return p


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/me")
def me(p: Principal = Depends(principal)) -> dict:
    return {"annotator": p.annotator, "split": p.split, "admin": p.admin}


@app.get("/sentences")
def sentences(p: Principal = Depends(principal)) -> dict:
    """The sentences THIS token is entitled to annotate.

    The hosted UI is deployed from the repo, where DATA/gold_standard is
    gitignored, so it has no sentence data of its own. Serving the sample here
    means the UI stores nothing, and sentence delivery is gated by the same
    token as writing — a practice token gets practice sentences, and the pilot
    sentences are not readable from the public UI URL without a token.
    """
    try:
        rows = load_rows(p.split)
    except SampleUnavailable as exc:
        # 503, not 404: the split is valid (it came from a trusted token), the
        # DEPLOYMENT is incomplete. A 404 would send the annotator looking for a
        # bad link when the researcher needs to redeploy.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"split": p.split, "n": len(rows), "rows": rows}


@app.get("/annotations")
def get_annotations(p: Principal = Depends(principal)) -> dict:
    return store().get_all(p.split, p.annotator)


@app.put("/annotations/{sentence_id}")
async def put_annotation(sentence_id: str, request: Request,
                         p: Principal = Depends(principal)) -> dict:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    store().put(p.split, p.annotator, sentence_id, payload)
    return {"saved": sentence_id}


@app.get("/admin/progress")
def admin_progress(split: str | None = None, p: Principal = Depends(admin)) -> dict:
    s = store()
    return {"splits": s.splits(), "progress": s.progress(split)}


@app.get("/admin/export.json")
def admin_export_json(split: str | None = None, p: Principal = Depends(admin)):
    return JSONResponse(store().export(split))


@app.get("/admin/export")
def admin_export(split: str | None = None, p: Principal = Depends(admin)):
    """ZIP laid out exactly like the local app's autosave files, so
    `python -m gold_standard.iaa --ann-dir <unzipped>` works unchanged."""
    s = store()
    data = s.export(split)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for sp, by_coder in data.items():
            for coder, anns in by_coder.items():
                z.writestr(f"{sp}_{coder}.json",
                           json.dumps(anns, indent=2, ensure_ascii=False))
        ev = s.events(split)
        if ev:
            sio = io.StringIO()
            w = csv.DictWriter(sio, fieldnames=list(ev[0].keys()))
            w.writeheader()
            w.writerows(ev)
            z.writestr("events.csv", sio.getvalue())
        z.writestr("progress.json", json.dumps(s.progress(split), indent=2))
    buf.seek(0)
    name = f"annotations_{split or 'all'}.zip"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})
