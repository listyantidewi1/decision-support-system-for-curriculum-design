r"""Mint annotator tokens into a gitignored file, and print the links to send.

Usage (from the repo root):
  .venv-ml\Scripts\python.exe -m annotation_server.make_tokens \
      --base-url https://your-host.example --split pilot_100 \
      --practice-split practice_10 --out api_keys/annotation_tokens.json

Writes the token table to --out (which MUST stay gitignored) and prints one
private link per annotator plus the admin link. Nothing is printed to any log.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gold_standard import ANNOTATORS

from .auth import mint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--split", default="pilot_100")
    ap.add_argument("--practice-split", default="practice_10")
    ap.add_argument("--out", default="api_keys/annotation_tokens.json")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists — refusing to overwrite live tokens. "
                         "Move it aside first if you really mean to re-mint.")
    table: dict[str, dict] = {}
    links: list[str] = []
    for code in ANNOTATORS:
        tok, entry = mint(code, args.split)
        table[tok] = entry
        links.append(f"  {code} (REAL {args.split}):     {args.base_url}/?token={tok}")
        ptok, pentry = mint(code, args.practice_split)
        table[ptok] = pentry
        links.append(f"  {code} (practice, discardable): {args.base_url}/?token={ptok}")
    atok, aentry = mint("ADMIN", args.split, admin=True)
    table[atok] = aentry

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"token table -> {out}  (KEEP PRIVATE; must stay gitignored)\n")
    print("Send each annotator ONLY their own two links:\n")
    print("\n".join(links))
    print(f"\n  ADMIN (yours, do not share):  {args.base_url}/admin?token={atok}")
    print("\nSet GS_TOKENS_FILE to that path on the host (or paste the file's "
          "contents into a GS_TOKENS secret).")


if __name__ == "__main__":
    main()
