"""OpenRouter entrypoint for SkillSpan-style annotation.

This is a compatibility wrapper around the existing annotation implementation.
Use this script in new commands so provider intent is clear.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("annotate_skillspan_with_jatevo.py")
    spec = importlib.util.spec_from_file_location("_skillspan_annotation_impl", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load annotation implementation: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv[0] = str(Path(__file__))
    raise SystemExit(module.main())
