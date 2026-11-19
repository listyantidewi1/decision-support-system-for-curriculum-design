"""Build lexicon v2 = lexicon v1 + technology names from O*NET.

Source: O*NET 30.2 Database, file "Technology Skills", U.S. Department of
Labor, Employment and Training Administration (USDOL/ETA), used under the
CC BY 4.0 licence. This script and the list it writes are a derived work:
the O*NET names were filtered and rewritten by the rules below, so the result
is NOT the O*NET list and must not be presented as one.

    python -m fivetag.build_lexicon_v2 DATA/taxonomies/onet/Technology_Skills_db_30_2.txt

Why rules are needed. O*NET writes "Amazon Web Services AWS software",
"Cascading style sheets CSS", "Atlassian JIRA". Job postings write "AWS",
"CSS", "Jira". Copying the O*NET strings would add entries that never match.

What is taken: the technologies O*NET marks "Hot Technology" or "In Demand"
for the computer occupations whose SOC code starts with 15-12 (software
developers, programmers, web developers, testers, database and systems roles).

Rules, applied to each O*NET name (every rule is recorded per entry in
`ONET_DERIVATION.tsv`, so any entry can be traced to its source name):

  0. names that describe a KIND of software, not a product, are left out
     ("Web application framework", "Operating system", "NoSQL"): the guideline
     tags generic groups as knowledge, never as a technology (rule 7.9);
  1. drop a trailing generic word: "software", "IDE";
  2. a trailing run of capitals whose letters occur, in order, in the words
     before it is an acronym: "Cascading style sheets CSS" gives "Cascading
     style sheets" and "CSS" (but "Microsoft Power BI" is left whole);
  3. a leading vendor name may be dropped: "Atlassian JIRA" gives "JIRA" too;
  4. a one-word name is matched CASE-SENSITIVELY (" !cs"), because short names
     are often ordinary words ("Chef", "Go", "Swift");
  5. skipped: one-letter names ("C", "R"), and names that are everyday words
     even when capitalised (the STOPLIST below).

Entries already in v1 (ignoring case) are not added again. Everything else in
v2 is a byte-for-byte copy of v1.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from .lexicon import FILES, LEXICON_ROOT, parse

ONET_VERSION = "30.2"
SOC_PREFIX = "15-12"
VENDORS = ("Amazon Web Services", "Amazon", "Apache", "Atlassian", "Google", "Microsoft",
           "Oracle", "IBM", "Adobe", "Apple", "Red Hat", "JetBrains", "HashiCorp", "Meta",
           "Facebook", "Mozilla", "Cisco", "SAP", "VMware", "Salesforce", "Eclipse", "GNU",
           "Progress", "Puppet Labs", "The MathWorks", "Unity Technologies", "Epic Games")
GENERIC_TAIL = ("software", "IDE")
# one-word names that are everyday words even with a capital letter
STOPLIST = {"access", "word", "office", "teams", "project", "docs", "sheets", "forms", "drive",
            "edge", "maps", "meet", "notes", "pages", "numbers", "keynote", "mail", "calendar",
            "photos", "explorer", "exchange", "dynamics", "express", "ant", "make", "chef",
            "puppet", "salt", "spring", "swift", "go", "dart", "rust", "ruby", "julia", "hive",
            "pig", "spark", "storm", "flink", "impala", "beam", "camel", "ignite", "atom",
            "brackets", "slack", "zoom", "facebook", "twitter", "linkedin", "youtube",
            "instagram", "tiktok", "pinterest", "snapchat", "whatsapp", "safari", "chrome",
            "firefox", "python", "analytics", "workspace", "power", "cloud", "database",
            "bi", "services", "erp", "cms", "gis", "sso", "ods", "mq", "xd", "illustrator",
            "web services"}
# O*NET names that are a kind of software or a concept, not a product (rule 0)
EXCLUDE = {"Content management systems CMS", "Firewall software", "NoSQL", "RESTful API",
           "Single sign-on SSO", "Operational Data Store ODS",
           "Geographic information system GIS software",
           "Geographic information system GIS systems",
           "Low-level virtual machine LLVM compilers", "JavaScript framework"}


def is_generic_kind(example: str) -> bool:
    """A sentence-case common-noun phrase: "Web application framework"."""
    words = example.split()
    if words and words[-1] == "software":
        words = words[:-1]
    if example in EXCLUDE or " ".join(words) in EXCLUDE:
        return True
    return len(words) >= 2 and all(w.islower() for w in words[1:]) and words[0][1:].islower()


def _is_acronym_of(acr: str, name: str) -> bool:
    letters = [c for c in acr.lower() if c.isalpha()]
    pos = 0
    low = name.lower()
    for c in letters:
        pos = low.find(c, pos)
        if pos < 0:
            return False
        pos += 1
    return len(letters) >= 2
_ACRONYM_TAIL = re.compile(r"^(?P<name>.+?)\s+(?P<acr>(?:[A-Z][A-Z0-9+#./\-]{1,7})(?:\s+[A-Z0-9]{1,4})?)$")


def derive(example: str) -> list[tuple[str, str]]:
    """(entry, rule) pairs for one O*NET example name."""
    out: list[tuple[str, str]] = []
    name = " ".join(example.split())
    for tail in GENERIC_TAIL:
        if name.endswith(" " + tail):
            name = name[: -len(tail) - 1].strip()
    variants = [(name, "name")]
    m = _ACRONYM_TAIL.match(name)
    if m and len(m.group("name").split()) >= 2 and _is_acronym_of(m.group("acr"), m.group("name")):
        variants = [(m.group("name"), "name without acronym"), (m.group("acr"), "acronym")]
    for text, rule in list(variants):
        for vendor in VENDORS:
            if text.startswith(vendor + " ") and len(text) > len(vendor) + 1:
                rest = text[len(vendor) + 1:].strip()
                variants.append((rest, f"{rule}, vendor dropped"))
                break
    seen = set()
    for text, rule in variants:
        text = text.strip(" ,;")
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        one_word = " " not in text
        if len(text) < 2:
            continue                                   # "C", "R"
        if key in STOPLIST:
            continue
        if not any(c.isalpha() for c in text):
            continue
        out.append((text + (" !cs" if one_word else ""), rule))
    return out


def read_onet(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    keep: dict[str, dict] = {}
    for r in rows:
        if not r["O*NET-SOC Code"].startswith(SOC_PREFIX):
            continue
        if r["Hot Technology"] != "Y" and r["In Demand"] != "Y":
            continue
        if is_generic_kind(r["Example"]):
            continue
        keep.setdefault(r["Example"], r)
    return [keep[k] for k in sorted(keep)]


def build(onet_path: Path, out_version: str = "v2", base_version: str = "v1",
          root: Path = LEXICON_ROOT) -> dict:
    base, out = root / base_version, root / out_version
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if name != "technologies":
            shutil.copyfile(base / f"{name}.txt", out / f"{name}.txt")
    base_text = (base / "technologies.txt").read_text(encoding="utf-8")
    have = {e.text.lower() for e in parse(base_text)[""]}
    added, trace, n_source = [], [], 0
    for row in read_onet(onet_path):
        n_source += 1
        for entry, rule in derive(row["Example"]):
            bare = entry[:-4] if entry.endswith(" !cs") else entry
            status = "added"
            if bare.lower() in have:
                status = "already in v1"
            else:
                have.add(bare.lower())
                added.append(entry)
            trace.append((row["Example"], row["Commodity Title"], entry, rule, status))
    header = (f"# --- added in {out_version}: derived from O*NET {ONET_VERSION} \"Technology Skills\"\n"
              f"# (USDOL/ETA, CC BY 4.0), SOC {SOC_PREFIX}xx, Hot Technology or In Demand.\n"
              f"# Rewritten by the rules in fivetag/build_lexicon_v2.py; see ONET_DERIVATION.tsv.\n")
    text = base_text.rstrip("\n") + "\n" + header + "\n".join(added) + "\n"
    (out / "technologies.txt").write_text(text, encoding="utf-8", newline="\n")
    with (out / "ONET_DERIVATION.tsv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("onet_example\tonet_commodity_title\tentry\trule\tstatus\n")
        for t in trace:
            fh.write("\t".join(t) + "\n")
    (out / "SOURCES.md").write_text(
        f"# Lexicon {out_version}\n\n"
        f"`{out_version}` is `{base_version}` plus technology names derived from the O*NET "
        f"{ONET_VERSION} Database, file *Technology Skills*, by the U.S. Department of Labor, "
        f"Employment and Training Administration (USDOL/ETA), used under the CC BY 4.0 "
        f"licence. The names were filtered (SOC {SOC_PREFIX}xx, Hot Technology or In Demand) "
        f"and rewritten by `fivetag/build_lexicon_v2.py`; this list has been modified from "
        f"the original and is not an O*NET product. `ONET_DERIVATION.tsv` traces every added "
        f"entry to the O*NET name it came from.\n\n"
        f"- O*NET names used: {n_source}\n- entries added to `technologies.txt`: {len(added)}\n"
        f"- all other files: identical to `{base_version}`\n", encoding="utf-8", newline="\n")
    return {"out": str(out), "n_onet_names": n_source, "n_added": len(added),
            "n_already_in_base": sum(1 for t in trace if t[4] != "added")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onet_file")
    ap.add_argument("--out-version", default="v2")
    args = ap.parse_args()
    print(build(Path(args.onet_file), args.out_version))


if __name__ == "__main__":
    main()
