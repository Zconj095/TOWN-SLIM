"""
config_loader.py ― Tiny helper for reading project‑wide YAML configs
Author: (c) 2025

Current duty: load LLM_config.yaml so that llm_engine.py ‑and any other
module‑ can share the same dictionary without duplicating YAML parsing code.

Expansion ready: if you add embed_store.yaml or ui_settings.yaml, just add a
`get_config("embed_store")` call and place Embed_store.yaml beside this file.

Public helpers
--------------
get_config("llm")           -> returns dict from LLM_config.yaml
reload_all()                -> clear cache and re‑parse every file on next call
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from typing import Dict, Any

import yaml  # PyYAML dependency already present

# ---------------------------------------------------------------------------
# Logging – prefix “CL”
# ---------------------------------------------------------------------------
logger = logging.getLogger("CL")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[CL] %(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(h)

# ---------------------------------------------------------------------------
# Internal cache  {name: dict}
# ---------------------------------------------------------------------------
_CFG_CACHE: Dict[str, Dict[str, Any]] = {}

# Default config mapping  name -> filename
_CFG_MAP = {
    "llm": "LLM_config.yaml",
    # Future: "embed_store": "embed_store.yaml",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    logger.debug("CL: Loading YAML %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_config(name: str) -> Dict[str, Any]:
    """
    Return config dict for given name (“llm”).
    Caches after first read; call reload_all() to force fresh parse.
    """
    if name in _CFG_CACHE:
        return _CFG_CACHE[name]

    if name not in _CFG_MAP:
        raise ValueError(f"Unknown config name: {name}")

    fname = _CFG_MAP[name]
    # Look first in current working dir, then package resources
    path = Path(fname)
    if not path.exists():
        try:
            # fallback: bundled default inside town_core/
            with importlib.resources.path(__package__, fname) as p:
                path = Path(p)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {fname}") from None

    _CFG_CACHE[name] = _load_yaml(path)
    logger.info("CL: Config '%s' loaded (source=%s)", name, path)
    return _CFG_CACHE[name]


def reload_all() -> None:
    """Clear cache; next get_config() call re‑loads files."""
    _CFG_CACHE.clear()
    logger.info("CL: All configs cleared for reload")


# ---------------------------------------------------------------------------
# CLI tiny test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_config("llm")
    print(cfg)
