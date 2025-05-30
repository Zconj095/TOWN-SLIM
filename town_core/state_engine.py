"""
state_engine.py ― Lightweight replacement for UPBGE’s MANAGER object
Author: (c) 2025

A singleton dataclass‑style container that every script can import instead of
doing manager["property"] look‑ups.  Keeps runtime state in one predictable
place and can be serialised if you reload a scene or hop to Unreal.

Usage
-----
from town_core.state_engine import GameState

gs = GameState()               # always the same instance
gs.npc_name = "ava_rockford"   # write
print(gs.conversation_stage)   # read

# Dump / restore (e.g., on level reload)
saved = gs.to_dict()
gs.load_dict(saved)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Logging (“SE” prefix to follow user convention)
# ---------------------------------------------------------------------------
logger = logging.getLogger("SE")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[SE] %(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(h)


# ---------------------------------------------------------------------------
# GameState singleton
# ---------------------------------------------------------------------------
class _StateMeta(type):
    """Metaclass to enforce a single global instance."""

    _instance: Optional["GameState"] = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class GameState(metaclass=_StateMeta):
    """
    Replace scattered MANAGER string‑key properties with simple attributes.
    Extend with new fields as needed; existing save files stay compatible
    because unknown keys are ignored on load.
    """

    # ---- default values --------------------------------------------------
    npc_name: str = "unknown"
    conversation_stage: int = 0
    player_line: str = ""
    location: str = "unknown"
    last_llm_ts: str = ""            # ISO string of last prompt time

    # ---- serialisation helpers ------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return shallow copy of public attrs (no underscore keys)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def load_dict(self, data: Dict[str, Any]) -> None:
        """Set attributes from dict, ignore unknown keys."""
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ---- convenience setters --------------------------------------------
    def step_stage(self, amount: int = 1) -> None:
        self.conversation_stage += amount

    def mark_llm_call(self) -> None:
        self.last_llm_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # ---- save / load to file (JSON) -------------------------------------
    def save_to_file(self, path: str | Path = "gamestate.json") -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("SE: State saved to %s", path)

    def load_from_file(self, path: str | Path = "gamestate.json") -> None:
        p = Path(path)
        if p.exists():
            self.load_dict(json.loads(p.read_text(encoding="utf-8")))
            logger.info("SE: State loaded from %s", path)
        else:
            logger.warning("SE: State file not found: %s", path)


# ---------------------------------------------------------------------------
# CLI sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gs = GameState()
    gs.npc_name = "ava_rockford"
    gs.player_line = "Hello!"
    gs.location = "tea_shop"
    gs.step_stage()

    print("DICT  →", gs.to_dict())
    gs.save_to_file("_state_test.json")

    # reload fresh instance
    gs2 = GameState()
    gs2.load_from_file("_state_test.json")
    print("LOADED→", gs2.to_dict())
