"""
llm_engine.py ― Centralised chat/embedding gateway for the TOWN “tablet‑core”
Author: 2025

• Prompt → LLM → Response (chat)
• Text  → Embeddings (embed)
• Auto‑logging to sequential llmIO_###.txt
• Agnostic to provider; reads LLM_config.yaml

LLM_config.yaml example
-----------------------
provider: "ollama"              # or "openai"
model: "gemma3:12b"             # chat model
embed_model: "mxbai-embed-large"  # optional, falls back to model
base_url: "http://localhost:11434"
max_retries: 3
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai        # OpenAI, Groq, Grok, etc.
import ollama        # Local Ollama daemon

# ---------------------------------------------------------------------------
# Logging (LE prefix)
# ---------------------------------------------------------------------------
logger = logging.getLogger("LE")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[LE] %(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# YAML helper
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------
class _BaseAdapter:
    def chat(self, prompt: str, **kw) -> str: ...
    def embed(self, texts: List[str]) -> List[List[float]]: ...


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------
class _OllamaAdapter(_BaseAdapter):
    def __init__(self, chat_model: str, embed_model: str):
        self.chat_model = chat_model
        self.embed_model = embed_model

    # CHAT
    def chat(self, prompt: str, **kw) -> str:
        resp = ollama.chat(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return resp["message"]["content"]

    # EMBED
    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs: List[List[float]] = []
        for txt in texts:
            resp = ollama.embeddings(model=self.embed_model, prompt=txt)
            vecs.append(resp["embedding"])
        return vecs


# ---------------------------------------------------------------------------
# OpenAI‑style adapter
# ---------------------------------------------------------------------------
class _OpenAIAdapter(_BaseAdapter):
    def __init__(self, cfg: Dict[str, Any], chat_model: str, embed_model: str):
        key = cfg.get("api_key", "")
        if key.startswith("ENV:"):
            key = os.getenv(key[4:], "")
        openai.api_key = key
        openai.base_url = cfg.get("base_url", "https://api.openai.com/v1")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = int(cfg.get("timeout", 60))

    def chat(self, prompt: str, **kw) -> str:
        response = openai.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
        )
        return response.choices[0].message.content

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = openai.embeddings.create(
            model=self.embed_model,
            input=texts,
            timeout=self.timeout,
        )
        return [d.embedding for d in response.data]


# ---------------------------------------------------------------------------
# LLMEngine singleton
# ---------------------------------------------------------------------------
class LLMEngine:
    _instance: Optional["LLMEngine"] = None
    _CFG_PATH = Path("LLM_config.yaml")

    # enforce singleton
    def __new__(cls, *a, **kw):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # INIT
    def __init__(self, cfg_path: Path | str | None = None):
        if hasattr(self, "_initd"):
            return
        self._initd = True

        self.cfg_path = Path(cfg_path or self._CFG_PATH)
        self._load_and_build()
        self.max_retries = int(self.cfg.get("max_retries", 3))

        self.io_dir = Path("llm_logs")
        self.io_dir.mkdir(exist_ok=True)

    # internal
    def _load_and_build(self):
        self.cfg = _load_yaml(self.cfg_path)
        provider = self.cfg["provider"].lower()

        chat_model = self.cfg["model"]
        embed_model = self.cfg.get("embed_model", chat_model)

        if provider == "ollama":
            logger.info("LE: Using Ollama (%s chat, %s embed)", chat_model, embed_model)
            self.adapter = _OllamaAdapter(chat_model, embed_model)
        elif provider == "openai":
            logger.info("LE: Using OpenAI‑style (%s chat, %s embed)", chat_model, embed_model)
            self.adapter = _OpenAIAdapter(self.cfg, chat_model, embed_model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # public
    def reload_config(self):
        logger.info("LE: Reloading config")
        self._load_and_build()

    # CHAT
    def chat(self, prompt: str, **kw) -> str:
        attempt = 0
        while True:
            try:
                reply = self.adapter.chat(prompt, **kw)
                self._log_io(prompt, reply)
                return reply
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                back = 2 ** attempt
                logger.warning("LE: retry %d in %ds (%s)", attempt, back, e)
                time.sleep(back)

    # EMBED
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.adapter.embed(texts)

    # IO log
    def _log_io(self, prompt: str, reply: str):
        files = sorted(self.io_dir.glob("llmIO_*.txt"))
        idx = int(files[-1].stem.split("_")[-1]) + 1 if files else 1
        f = self.io_dir / f"llmIO_{idx:03}.txt"
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("=== PROMPT ===\n" + prompt + "\n\n=== RESPONSE ===\n" + reply)
        logger.info("LE: Logged exchange to %s", f)


# ---------------------------------------------------------------------------
# CLI smoke‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python llm_engine.py \"prompt\"")
        sys.exit(1)
    print(LLMEngine().chat(sys.argv[1]))
