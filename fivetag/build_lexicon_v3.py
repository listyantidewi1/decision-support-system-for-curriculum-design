"""Build lexicon v3 = lexicon v2 + the ICT branch of ESCO.

Source: ESCO, the European classification of Skills, Competences,
Qualifications and Occupations, published by the European Commission
(Directorate-General for Employment, Social Affairs and Inclusion), read
through its public web-service API (https://ec.europa.eu/esco/api). ESCO may be
reused with attribution; this list is a derived work, changed by the rules
below, and is not an ESCO product. The ESCO version served by the API on the
day of the download is recorded in SOURCES.md.

    python -m fivetag.build_lexicon_v3 fetch    # about 600 polite requests, cached
    python -m fivetag.build_lexicon_v3 build

What is taken: every knowledge concept under ISCED-F 061, "information and
communication technologies (ICTs)" — computer use, database and network design
and administration, software and applications development and analysis, and
the two residual groups.

ESCO lists two different kinds of thing there, and the annotation guideline
tags them differently, so each ESCO concept is sorted into one of two lists:

  technology (TT)  a named product, language, tool or standard:
                   "Python (computer programming)", "Oracle Relational Database".
                   The bracketed qualifier is dropped; English alternative labels
                   that are themselves names are added.
  knowledge (KN)   a concept or field written as a common-noun phrase:
                   "database quality standards", "ICT network routing".
                   Added as a knowledge term. One-word concepts are left out as
                   too general.

The sorting rule is mechanical (does the name contain a proper-name token?) and
every decision is written to ESCO_DERIVATION.tsv, so it can be audited and
corrected. It has NOT been reviewed by an expert.

`ESCO_ICT.tsv` keeps, for every concept, its ESCO URI and its parent group. The
time analysis uses the parent group to move a volatile technology up to a more
stable node (proposal §3.3.4, "penjenjangan abstraksi").
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .build_lexicon_v2 import STOPLIST
from .lexicon import FILES, LEXICON_ROOT, parse

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "DATA" / "taxonomies" / "esco"
API = "https://ec.europa.eu/esco/api/resource/{kind}?uri={uri}&language=en"
ROOT_URI = "http://data.europa.eu/esco/isced-f/061"
PAUSE_SECONDS = 0.4
# capitalised tokens that do not make a name: they open a common-noun phrase
NOT_A_NAME = {"ICT", "ICTs", "IT", "API", "APIs", "CAD", "CAE", "AI", "UI", "UX", "QA", "DB"}
GENERIC_NAMES = {"nosql", "internet", "web", "cloud", "blockchain", "devops", "iot",
                 "internet of things", "assembly", "canvas", "absorb", "cisco", "javax",
                 "cisco systems", "ml", "saas", "less", "computer science", "scratch"}
# a phrase that ends in one of these names a KIND of thing, whatever its first word
GENERIC_HEADS = {"software", "softwares", "system", "systems", "tools", "tool", "technologies",
                 "technology", "operations", "specifications", "framework", "frameworks",
                 "languages", "language", "methods", "techniques", "standards", "platforms"}
MAX_ALT_WORDS = 3
_PAREN = re.compile(r"\s*\(([^)]*)\)\s*$")


def _get(kind: str, uri: str, refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", uri)[-120:] + ".json")
    if path.is_file() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    url = API.format(kind=kind, uri=urllib.parse.quote(uri, safe=""))
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "research-taxonomy-client"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(PAUSE_SECONDS)
    return data


def fetch(log=print) -> list[dict]:
    """Every skill concept under ISCED-F 061, with its parent group."""
    root = _get("concept", ROOT_URI)
    rows = []
    for group in root["_links"].get("narrowerConcept", []):
        g = _get("concept", group["uri"])
        skills = g["_links"].get("narrowerSkill", [])
        log(f"{group['uri'].rsplit('/', 1)[1]}  {g['title']}: {len(skills)} concepts")
        for link in skills:
            s = _get("skill", link["uri"])
            kinds = [t.get("title", "") for t in s["_links"].get("hasSkillType", [])]
            rows.append({"uri": s["uri"], "title": s["title"],
                         "alt_en": (s.get("alternativeLabel") or {}).get("en") or [],
                         "skill_type": kinds[0] if kinds else "",
                         "parent_code": group["uri"].rsplit("/", 1)[1],
                         "parent_title": g["title"]})
    (CACHE / "esco_ict.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    return rows


def split_title(title: str) -> tuple[str, str]:
    """('Python', 'computer programming') from 'Python (computer programming)'."""
    m = _PAREN.search(title)
    return (title[:m.start()].strip(), m.group(1).strip()) if m else (title.strip(), "")


def is_name(text: str) -> bool:
    """Does the phrase contain a proper-name token? A capital that is not the
    first letter of the phrase, an inner capital or digit ('JavaScript', 'DB2'),
    or a single capitalised word ('Python')."""
    words = text.split()
    if not words or text.lower() in GENERIC_NAMES:
        return False
    if len(words) >= 2 and words[-1].lower().strip(",.") in GENERIC_HEADS:
        return False
    if len(words) == 1:
        w = words[0]
        return w.lower() not in GENERIC_NAMES and w not in NOT_A_NAME and (
            w[0].isupper() or any(c.isupper() or c.isdigit() for c in w[1:]))
    for i, w in enumerate(words):
        if w in NOT_A_NAME or w.strip(",.").lower() in GENERIC_NAMES:
            continue
        core = w.strip("(),.")
        if not core:
            continue
        if any(c.isupper() or c.isdigit() for c in core[1:]):
            return True
        if i > 0 and core[0].isupper():
            return True
    return False


def derive(row: dict) -> list[tuple[str, str, str]]:
    """(list, entry, rule) triples for one ESCO concept."""
    name, _qualifier = split_title(row["title"])
    out: list[tuple[str, str, str]] = []
    if is_name(name):
        for text, rule in [(name, "title")] + [(a, "alternative label") for a in row["alt_en"]]:
            text = split_title(" ".join(text.split()))[0]
            if len(text) < 2 or text.lower() in STOPLIST or text.lower() in GENERIC_NAMES:
                continue
            if rule != "title" and (not is_name(text) or len(text.split()) > MAX_ALT_WORDS):
                continue
            if len(text.split()) > 5:
                continue
            out.append(("technology", text + (" !cs" if " " not in text else ""), rule))
    elif 2 <= len(name.split()) <= 6 and name.lower() not in GENERIC_NAMES:
        out.append(("knowledge", name[0].lower() + name[1:] if name.split()[0] not in NOT_A_NAME
                    else name, "title, common-noun phrase"))
    return out


def build(out_version: str = "v3", base_version: str = "v2", root: Path = LEXICON_ROOT,
          rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else json.loads(
        (CACHE / "esco_ict.json").read_text(encoding="utf-8"))
    base, out = root / base_version, root / out_version
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if name not in ("technologies", "knowledge"):
            shutil.copyfile(base / f"{name}.txt", out / f"{name}.txt")
    for extra in ("ONET_DERIVATION.tsv",):
        if (base / extra).is_file():
            shutil.copyfile(base / extra, out / extra)
    tech_text = (base / "technologies.txt").read_text(encoding="utf-8")
    kn_text = (base / "knowledge.txt").read_text(encoding="utf-8")
    have_tt = {e.text.lower() for e in parse(tech_text)[""]}
    kn = parse(kn_text)
    have_kn = {e.text.lower() for sec in ("terms", "groups") for e in kn.get(sec, [])}
    add_tt, add_kn, trace = [], [], []
    for row in sorted(rows, key=lambda r: r["title"].lower()):
        derived = derive(row)
        if not derived:
            trace.append((row["uri"], row["title"], row["parent_code"], "", "", "left out"))
        for which, entry, rule in derived:
            bare = (entry[:-4] if entry.endswith(" !cs") else entry).lower()
            have = have_tt if which == "technology" else have_kn
            status = "already present" if bare in have or bare in (have_tt | have_kn) else "added"
            if status == "added":
                have.add(bare)
                (add_tt if which == "technology" else add_kn).append(entry)
            trace.append((row["uri"], row["title"], row["parent_code"], which, entry,
                          f"{rule}; {status}"))
    note = (f"derived from ESCO (European Commission), ISCED-F 061 ICT branch, via the\n"
            f"# public ESCO API. Rewritten by fivetag/build_lexicon_v3.py; see ESCO_DERIVATION.tsv.\n")
    (out / "technologies.txt").write_text(
        tech_text.rstrip("\n") + f"\n# --- added in {out_version}: " + note + "\n".join(add_tt) + "\n",
        encoding="utf-8", newline="\n")
    kn_text = kn_text.replace(chr(13) + chr(10), chr(10))
    marker = chr(10) + "[groups]" + chr(10)     # the section header line, not its mention in the comment
    assert kn_text.count(marker) == 1
    head, tail = kn_text.split(marker)
    marker = "[groups]" + chr(10)
    (out / "knowledge.txt").write_text(
        head.rstrip("\n") + f"\n# --- added in {out_version} to [terms]: " + note
        + "\n".join(add_kn) + "\n" + marker + tail, encoding="utf-8", newline="\n")
    with (out / "ESCO_DERIVATION.tsv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("esco_uri\tesco_title\tparent_code\tlist\tentry\trule_and_status\n")
        for t in trace:
            fh.write("\t".join(t) + "\n")
    with (out / "ESCO_ICT.tsv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("esco_uri\tesco_title\tname\tkind\tparent_code\tparent_title\n")
        for r in sorted(rows, key=lambda r: r["title"].lower()):
            name = split_title(r["title"])[0]
            fh.write("\t".join([r["uri"], r["title"], name,
                                "technology" if is_name(name) else "knowledge",
                                r["parent_code"], r["parent_title"]]) + "\n")
    base_sources = (base / "SOURCES.md").read_text(encoding="utf-8") if (base / "SOURCES.md").is_file() else ""
    (out / "SOURCES.md").write_text(
        f"# Lexicon {out_version}\n\n"
        f"`{out_version}` is `{base_version}` plus entries derived from **ESCO**, the European "
        f"classification of Skills, Competences, Qualifications and Occupations, published by "
        f"the European Commission. The ICT branch (ISCED-F 061, {len(rows)} concepts) was read "
        f"through the public ESCO API on {time.strftime('%Y-%m-%d')}. The concepts were sorted "
        f"into technology names and knowledge terms and rewritten by "
        f"`fivetag/build_lexicon_v3.py`; this list has been modified from the original and is "
        f"not an ESCO product. `ESCO_DERIVATION.tsv` traces every entry to its ESCO concept; "
        f"`ESCO_ICT.tsv` keeps each concept's URI and parent group.\n\n"
        f"- technology entries added: {len(add_tt)}\n- knowledge terms added: {len(add_kn)}\n\n"
        f"---\n\nInherited from `{base_version}`:\n\n" + base_sources,
        encoding="utf-8", newline="\n")
    return {"out": str(out), "n_esco_concepts": len(rows), "n_technology_added": len(add_tt),
            "n_knowledge_added": len(add_kn)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "build"])
    args = ap.parse_args()
    if args.cmd == "fetch":
        rows = fetch()
        print(f"{len(rows)} concepts -> {CACHE / 'esco_ict.json'}")
    else:
        print(build())


if __name__ == "__main__":
    main()
