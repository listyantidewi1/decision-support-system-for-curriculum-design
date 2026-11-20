"""What the two dashboards share: where things are, how they are read, colours."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Read at call time, so a test (or a deployment) can point the dashboards
    at another folder with COMPETENCY_DATA_ROOT."""
    return Path(os.environ.get("COMPETENCY_DATA_ROOT", REPO_ROOT / "DATA" / "competency"))


def profiles_dir() -> Path:
    return data_root() / "profiles"


def runs_dir() -> Path:
    return data_root() / "runs"


STATUS_LABEL = {"eksploratif": "Eksploratif", "kandidat_tinjauan": "Kandidat tinjauan",
                "siap_ditelaah_formal": "Siap ditelaah formal"}
STATUS_ORDER = ["siap_ditelaah_formal", "kandidat_tinjauan", "eksploratif"]
COVERAGE_LABEL = {"tercakup_jelas": "Tercakup jelas", "tercakup_sebagian": "Tercakup sebagian",
                  "belum_tercakup": "Belum tercakup",
                  "masukan_pembaruan": "Berpotensi menjadi masukan pembaruan"}
DELTA_LABEL = {"menambah": "Menambah", "memperdalam": "Memperdalam",
               "meninjau_ulang_jenjang": "Meninjau ulang jenjang",
               "mempertahankan": "Mempertahankan", "mengurangi_penekanan": "Mengurangi penekanan",
               "pemantauan": "Pemantauan saja"}
BLUE = "#2a78d6"
LEVEL_LABEL = {"remember": "mengingat", "understand": "memahami", "apply": "menerapkan",
               "analyze": "menganalisis", "evaluate": "mengevaluasi", "create": "mencipta"}


LEVEL_CODES = ["C6", "C5", "C4", "C3", "C2", "C1"]
NO_LEVEL = "tanpa kata kerja"


def level_text(cog: dict | None) -> str:
    """'C6 mencipta' for a level read from the postings, '' when there is none."""
    if not cog or not cog.get("level"):
        return ""
    return f"{cog['code']} {LEVEL_LABEL.get(cog['level'], cog['level'])}"


def statement_text(rec: dict) -> str:
    """The statement a reader sees: the SKKNI-style formulation when the
    profile has one, else the rule statement of the pipeline."""
    f = rec.get("formulated") or {}
    return f.get("pernyataan") or rec["statement"]["text"]


def level_code(rec: dict) -> str | None:
    """The filter value of a recommendation: its Bloom code, NO_LEVEL when its
    postings use no verb, None for a profile published before levels existed."""
    cog = rec.get("cognitive")
    if cog is None:
        return None
    return cog.get("code") or NO_LEVEL


def list_profiles() -> list[dict]:
    """Every published profile, releases first, newest version first."""
    out = []
    for kind in ("releases", "drafts"):
        folder = profiles_dir() / kind
        if folder.is_dir():
            for d in sorted(folder.iterdir(), reverse=True):
                if (d / "profile.json").is_file():
                    out.append({"kind": kind, "version": d.name, "path": d / "profile.json",
                                "label": f"{d.name}  ({'rilis' if kind == 'releases' else 'draf'})"})
    return out


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_runs() -> list[Path]:
    if not runs_dir().is_dir():
        return []
    return sorted((d for d in runs_dir().iterdir() if (d / "report.json").is_file()),
                  key=lambda d: d.stat().st_mtime, reverse=True)


def counts(rows, key) -> dict:
    out: dict = {}
    for r in rows:
        k = key(r)
        if k is not None and k != "":
            out[k] = out.get(k, 0) + 1
    return out
