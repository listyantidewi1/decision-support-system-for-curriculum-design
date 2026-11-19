"""Versioned lexicons: plain text files, loaded once, identified by a hash.

The proposal requires that "seluruh konfigurasi, versi model, prompt, kamus, dan
ambang didokumentasikan dalam registri konfigurasi". A lexicon version is a
folder under `fivetag/lexicons/`; its identity is the SHA-256 of every file in
it. Editing a file changes the hash, so a result can always be tied to the exact
word lists that produced it. To change the lists after results exist, copy the
folder to a new version instead of editing in place.

File format: one entry per line; `#` starts a comment line; `[name]` starts a
section; an entry ending in ` !cs` is matched case-sensitively.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

LEXICON_ROOT = Path(__file__).resolve().parent / "lexicons"
DEFAULT_VERSION = "v1"
CS_MARK = " !cs"

FILES = ("technologies", "soft_skills", "work_behaviors", "knowledge", "actions", "boundaries")


@dataclass(frozen=True)
class Entry:
    text: str
    case_sensitive: bool = False


def parse(text: str) -> dict[str, list[Entry]]:
    """Section name -> entries. Lines before any `[section]` go under ''."""
    out: dict[str, list[Entry]] = {"": []}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            out.setdefault(section, [])
            continue
        cs = raw.rstrip().endswith(CS_MARK)
        if cs:
            line = raw.rstrip()[: -len(CS_MARK)].strip()
        out[section].append(Entry(line, cs))
    return out


def compile_entries(entries: list[Entry]) -> list[re.Pattern]:
    """Up to two alternation patterns (case-insensitive, case-sensitive), each
    listing longer entries first so the longest entry wins at any position.

    An entry matches only when it is not glued to a letter or digit on either
    side, so `Java` does not fire inside `JavaScript` and `Go` not inside `Golang`.
    Spaces in an entry match any run of whitespace.
    """
    def alt(items: list[Entry]) -> str:
        parts = sorted({e.text for e in items}, key=lambda s: (-len(s), s))
        return "|".join(r"\s+".join(re.escape(w) for w in p.split()) for p in parts)

    pats = []
    ci = [e for e in entries if not e.case_sensitive]
    cs = [e for e in entries if e.case_sensitive]
    for items, flags in ((ci, re.IGNORECASE), (cs, 0)):
        if items:
            pats.append(re.compile(r"(?<![A-Za-z0-9])(?:" + alt(items) + r")(?![A-Za-z0-9])", flags))
    return pats


@dataclass(frozen=True)
class Lexicon:
    version: str
    sha256: str
    sections: dict = field(hash=False, compare=False, default_factory=dict)   # (file, section) -> entries

    def entries(self, file: str, section: str = "") -> list[Entry]:
        return list(self.sections.get((file, section), []))

    def words(self, file: str, section: str = "") -> set[str]:
        """Lower-cased entry texts of a section."""
        return {e.text.lower() for e in self.entries(file, section)}

    def lines(self, file: str, section: str) -> list[list[str]]:
        """Each entry split on whitespace, lower-cased: for 'one verb per line' sections."""
        return [e.text.lower().split() for e in self.entries(file, section)]

    def patterns(self, file: str, section: str = "") -> list[re.Pattern]:
        return compile_entries(self.entries(file, section))

    def manifest(self) -> dict:
        return {"lexicon_version": self.version, "lexicon_sha256": self.sha256,
                "n_entries": {f"{f}:{s}" if s else f: len(v)
                              for (f, s), v in sorted(self.sections.items()) if v}}


def lexicon_sha256(folder: Path) -> str:
    """Hash of every lexicon file, independent of the platform's line endings."""
    h = hashlib.sha256()
    for name in FILES:
        data = (folder / f"{name}.txt").read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode() + b"\0" + data + b"\0")
    return h.hexdigest()


@lru_cache(maxsize=None)
def load(version: str = DEFAULT_VERSION, root: Path | None = None) -> Lexicon:
    folder = (root or LEXICON_ROOT) / version
    if not folder.is_dir():
        raise FileNotFoundError(f"no lexicon version {version!r} under {root or LEXICON_ROOT}")
    sections: dict[tuple[str, str], list[Entry]] = {}
    for name in FILES:
        path = folder / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"lexicon {version} is missing {path.name}")
        for section, entries in parse(path.read_text(encoding="utf-8")).items():
            sections[(name, section)] = entries
    return Lexicon(version, lexicon_sha256(folder), sections)
