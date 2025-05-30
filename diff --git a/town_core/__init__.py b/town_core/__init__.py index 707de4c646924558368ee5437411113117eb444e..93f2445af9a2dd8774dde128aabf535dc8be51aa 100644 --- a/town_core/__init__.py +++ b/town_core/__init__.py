diff --git a/town_core/__init__.py b/town_core/__init__.py
index 707de4c646924558368ee5437411113117eb444e..93f2445af9a2dd8774dde128aabf535dc8be51aa 100644
--- a/town_core/__init__.py
+++ b/town_core/__init__.py
@@ -1,30 +1,32 @@
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
+from .recall_manager import RecallManager
 
 __all__ = [
     "LLMEngine",
     "EmbedEngine",
     "MemoryEngine",
     "PromptEngine",
     "GameState",
+    "RecallManager",
 ]
