"""Token-scoped access for the annotation collector.

Each annotator gets an opaque bearer token that maps to exactly one
(annotator_code, split). This replaces the `?annotator=AV` query parameter the
local app used, which was guessable: anyone with the URL could open any
annotator's workspace, and a curious click became indistinguishable from real
expert annotation. A real contamination of exactly this kind was found in the
pilot (an interface trial had left 8 judged sentences in AV's file).

The split in the token is also the real/practice separation: a practice token
points at a practice split, so warm-up clicks can never land in pilot_100.

Tokens are NEVER committed. They are supplied as:
  * env GS_TOKENS — a JSON object {token: {"annotator": "AV", "split": "pilot_100"}}
  * or env GS_TOKENS_FILE — path to that JSON (gitignored)
An entry with "admin": true may read every annotator and download exports.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Principal:
    token_id: str          # first 8 chars only — safe to log
    annotator: str
    split: str
    admin: bool = False


def _raw_tokens() -> dict:
    blob = os.environ.get("GS_TOKENS")
    if blob:
        return json.loads(blob)
    path = os.environ.get("GS_TOKENS_FILE")
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def resolve(token: str | None) -> Principal | None:
    """Constant-time-ish lookup of a bearer token; None when unknown."""
    if not token:
        return None
    table = _raw_tokens()
    # compare_digest against every key so a wrong token costs the same as a right
    # one regardless of position, and never short-circuits on a prefix match
    match = None
    for known, meta in table.items():
        if secrets.compare_digest(str(known), str(token)):
            match = (known, meta)
    if match is None:
        return None
    known, meta = match
    return Principal(token_id=str(known)[:8], annotator=str(meta.get("annotator", "")),
                     split=str(meta.get("split", "")), admin=bool(meta.get("admin")))


def mint(annotator: str, split: str, admin: bool = False) -> tuple[str, dict]:
    """Generate one token entry. Caller is responsible for storing it safely."""
    tok = secrets.token_urlsafe(32)
    entry = {"annotator": annotator, "split": split}
    if admin:
        entry["admin"] = True
    return tok, entry
