"""
prompt_engine.py ― Jinja2 template renderer for every prompt & summary
=====================================================================
Author: 2025

• Searches for *.jinja templates in town_core/prompts/
• Keeps a single compiled Jinja Environment for performance
• Main entry point:
      PromptEngine.render("npc_chat", **slots)  → str

Expected slot names for npc_chat.jinja (template must reference these)
---------------------------------------------------------------------
npc_name              : str            – name of the NPC speaking
system_rules          : str            – full text from FULL_SYSTEM_PROMPT.txt
assistant_rules       : str            – full text from FULL_ASSISTANT_PROMPT.txt
system_chunks         : list[str]      – static lore snippets (top‑K)
sheet_chunks          : list[str]      – character sheet snippets (top‑K)
memory_chunks         : list[str]      – session memory summaries (top‑K)
conversation_history  : list[str]      – last H raw dialogue lines
player_line           : str            – most recent user utterance
instructions          : str            – extra guidance (“Respond in character.”)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

# ---------------------------------------------------------------------------
# Logging (PE prefix)
# ---------------------------------------------------------------------------
logger = logging.getLogger("PE")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[PE] %(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# PromptEngine (singleton)
# ---------------------------------------------------------------------------
class PromptEngine:
    """
    Lightweight facade around Jinja2; ensures templates are compiled once.

    Example
    -------
    pe = PromptEngine()
    txt = pe.render(
        "npc_chat",
        npc_name="Ava",
        system_rules=open("FULL_SYSTEM_PROMPT.txt").read(),
        assistant_rules=open("FULL_ASSISTANT_PROMPT.txt").read(),
        system_chunks=["[lore‑1]", "[lore‑2]"],
        sheet_chunks=["[trait‑1]"],
        memory_chunks=["[summary‑1]", "[summary‑2]"],
        conversation_history=["PLAYER: Hi", "AVA: Hello."],
        player_line="How are you?",
        instructions="Stay in character.",
    )
    """

    _instance = None  # single shared engine

    def __new__(cls, templates_path: str | Path | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Initialise Jinja environment the first time the singleton is used
    # ------------------------------------------------------------------
    def __init__(self, templates_path: str | Path | None = None):
        if hasattr(self, "_initialized"):
            return  # already initialised on prior instantiation
        self._initialized = True

        self.templates_path = Path(templates_path or Path(__file__).parent / "prompts")
        if not self.templates_path.exists():
            raise FileNotFoundError(f"Prompt templates folder not found: {self.templates_path}")

        # No HTML output → disable auto‑escaping entirely
        self.env = Environment(
            loader=FileSystemLoader(self.templates_path),
            autoescape=False,
            trim_blocks=True,    # drop leading newline inside blocks
            lstrip_blocks=True,  # strip indentation before blocks
        )

        # Handy filter: join list items with newline bullets
        self.env.filters["bullets"] = lambda seq: "\n".join(f"- {s}" for s in seq)

        logger.info("PE: Loaded templates from %s", self.templates_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(self, template_name: str, **slots: Dict[str, Any]) -> str:
        """
        Render *template_name*.jinja with the supplied keyword slots.
        Raises TemplateNotFound if the file is missing.
        """
        try:
            template = self.env.get_template(f"{template_name}.jinja")
        except TemplateNotFound as exc:
            logger.error("PE: Template '%s' not found", template_name)
            raise exc

        text = template.render(**slots)
        logger.debug("PE: Rendered %s (len=%d chars)", template_name, len(text))
        return text

    # Convenience: list available templates (without .jinja suffix)
    def available_templates(self) -> list[str]:
        return [
            p.stem
            for p in self.templates_path.glob("*.jinja")
            if p.is_file() and not p.name.startswith("_")
        ]


# ---------------------------------------------------------------------------
# CLI test – renders a template with placeholder data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    pe = PromptEngine()
    if len(sys.argv) < 2:
        print("Usage: python prompt_engine.py <template_name>")
        print("Available:", ", ".join(pe.available_templates()))
        sys.exit(1)

    tpl_name = sys.argv[1]

    dummy = pe.render(
        tpl_name,
        npc_name="Ava",
        system_rules="### SYSTEM RULES\nBe friendly.",
        assistant_rules="### ASSISTANT RULES\nAlways greet politely.",
        system_chunks=["Edge‑of‑Town lies amid misty cliffs."],
        sheet_chunks=["Ava loves oolong tea."],
        memory_chunks=["Player tasted a new brew."],
        conversation_history=["PLAYER: Hi", "AVA: Welcome back!"],
        player_line="Where can I find the healer?",
        instructions="Respond in character.",
    )
    print(dummy)
