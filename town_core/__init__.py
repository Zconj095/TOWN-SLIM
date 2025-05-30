"""
town_core package initializer
-----------------------------

Convenience re‑exports so caller code can simply do:

    import town_core as tc

    llm  = tc.LLMEngine()
    mem  = tc.MemoryEngine()
    gs   = tc.GameState()
    txt  = tc.PromptEngine().render("npc_chat", ...)

Nothing else is executed at import-time; each sub-engine remains a lazy
singleton initialized on first use.
"""

from .llm_engine import LLMEngine
from .embed_engine import EmbedEngine
from .memory_engine import MemoryEngine
from .prompt_engine import PromptEngine
from .state_engine import GameState

__all__ = [
    "LLMEngine",
    "EmbedEngine",
    "MemoryEngine",
    "PromptEngine",
    "GameState",
]
