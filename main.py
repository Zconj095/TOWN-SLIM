"""
main.py — Thin CLI orchestrator for the TOWN “tablet‑core” demo
----------------------------------------------------------------
• Usage:  python main.py <npc_name>
• Type “quit” to exit.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from town_core.state_engine import GameState
from town_core.memory_engine import MemoryEngine
from town_core.prompt_engine import PromptEngine
from town_core.llm_engine import LLMEngine

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s [%(asctime)s]")

# ----------------------------------------------------------------------------
# Template name
# ----------------------------------------------------------------------------
NPC_CHAT_TEMPLATE = "npc_chat"

# ----------------------------------------------------------------------------
# Helper: create placeholder sheet on first run
# ----------------------------------------------------------------------------
def bootstrap_npc(npc_name: str, mem: MemoryEngine) -> None:
    sheet_path = Path(f"npc_data/{npc_name}/character_sheet.txt")
    if sheet_path.exists():
        return
    text = (
        f"{npc_name.replace('_', ' ').title()} is a placeholder NPC for the "
        "lean‑core demo.  Provide a richer sheet later."
    )
    mem.ensure_sheet(npc_name, text)
    logging.info(f"[Setup] Created default sheet for {npc_name}")

# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python main.py <npc_name>")
        sys.exit(1)
    npc_name = sys.argv[1].lower()

    # ---------- singletons ----------
    gs = GameState()
    mem = MemoryEngine()
    pe = PromptEngine()
    llm = LLMEngine()

    # ---------- static rules ----------
    FULL_SYS = Path("FULL_SYSTEM_PROMPT.txt").read_text(encoding="utf-8")
    FULL_ASST = Path("FULL_ASSISTANT_PROMPT.txt").read_text(encoding="utf-8")

    # ---------- initial state ----------
    gs.npc_name = npc_name
    gs.location = "default"
    bootstrap_npc(npc_name, mem)

    # path for rolling history file
    hist_path = Path(f"npc_data/{npc_name}/conversation_history.txt")
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"--- TALKING TO {npc_name.upper()} (type 'quit' to exit) ---")

    while True:
        player_line = input("YOU > ").strip()
        if player_line.lower() in {"quit", "exit"}:
            logging.info("Exiting…")
            break
        if not player_line:
            continue

        gs.player_line = player_line
        gs.step_stage()

        # 1) gather context (includes conversation_history and metrics)
        ctx = mem.fetch_context(npc_name, k_per_bucket=4)

        # 2) build prompt
        prompt = pe.render(
            NPC_CHAT_TEMPLATE,
            npc_name=npc_name,
            system_rules=FULL_SYS,
            assistant_rules=FULL_ASST,
            system_chunks=ctx["static_chunks"],
            sheet_chunks=ctx["sheet_chunks"],
            memory_chunks=ctx["memory_chunks"],
            conversation_history=ctx["conversation_history"],
            player_line=player_line,
            metrics=ctx["metrics"],
            instructions="Respond in character, adjusting tone based on metrics. Propose metric changes.",
        )

        # 3) LLM call
        reply = llm.chat(prompt)
        gs.mark_llm_call()

        # 4) update metrics from response
        mem.update_metrics_from_response(npc_name, reply)

        # 5) output
        print(f"{npc_name.upper()} > {reply.strip()}\n")

        # 6) append to rolling history
        with hist_path.open("a", encoding="utf-8") as fh:
            fh.write(f"PLAYER: {player_line}\n{npc_name.upper()}: {reply.strip()}\n")

        # 7) check history length and summarize if needed
        if hist_path.exists():
            history = hist_path.read_text(encoding="utf-8")
            if len(history) > mem.HISTORY_WINDOW:
                mem.summarize_history(npc_name, history)

if __name__ == "__main__":
    main()