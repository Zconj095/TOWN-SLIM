from __future__ import annotations
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import time
from town_core.embed_engine import EmbedEngine
from town_core.llm_engine import LLMEngine
from town_core.prompt_engine import PromptEngine
import ollama
import yaml

logger = logging.getLogger("ME")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s [%(asctime)s]"))
    logger.addHandler(_h)

class MemoryEngine:
    HISTORY_WINDOW = 5000
    KEEP_CHARS = 1000
    EMBED_RETRY_COUNT = 3
    EMBED_RETRY_DELAY = 1

    def __init__(self, data_dir: str | Path = "npc_data", lore_file: str | Path = "npc_data/dragonwarp_lore.jsonl"):
        self.data_dir = Path(data_dir)
        self.embed = EmbedEngine()
        self.llm = LLMEngine()
        self.pe = PromptEngine()
        self._static_domain = "static_lore"
        self._tssf_file = self.data_dir / "THE_STORY_SO_FAR.jsonl"
        self._summary_hashes = set()
        self._metrics_config = {}
        lore_path = Path(lore_file)
        if lore_path.exists():
            with lore_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        self.embed.put(text, self._static_domain)
        if self._tssf_file.exists():
            with self._tssf_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        npc = obj.get("npc", "")
                        if text and npc:
                            self._summary_hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
                            for attempt in range(self.EMBED_RETRY_COUNT):
                                try:
                                    result = ollama.embeddings(model="mxbai-embed-large", prompt=text)
                                    obj["embedding"] = result["embedding"]
                                    self.embed.put(text, f"{npc}_memory", npc=npc)
                                    break
                                except Exception as e:
                                    logger.warning("ME: Embed attempt %d failed for %s: %s", attempt + 1, npc, str(e))
                                    if attempt < self.EMBED_RETRY_COUNT - 1:
                                        time.sleep(self.EMBED_RETRY_DELAY)
                                    else:
                                        logger.error("ME: Failed to embed TSSF entry for %s after %d attempts", npc, self.EMBED_RETRY_COUNT)
                    except json.JSONDecodeError:
                        logger.warning("ME: Invalid JSON in THE_STORY_SO_FAR.jsonl: %s", line)

    def _npc_dir(self, npc: str) -> Path:
        d = self.data_dir / npc
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _history_path(self, npc: str) -> Path:
        return self._npc_dir(npc) / "conversation_history.txt"

    def ensure_sheet(self, npc: str, sheet_text: str) -> str:
        sheet_path = self._npc_dir(npc) / "character_sheet.txt"
        if not sheet_path.exists():
            sheet_path.write_text(sheet_text, encoding="utf-8")
        return self.embed.put(sheet_text, f"{npc}_sheet", npc=npc)

    def load_metrics(self, npc: str) -> Dict[str, float]:
        metrics_path = self._npc_dir(npc) / "npc_metrics.yaml"
        metrics = {"trust": -4.0, "affection": 0.0, "respect": -2.0}  # Fallback defaults
        config = {}
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            for key in ["trust", "affection", "respect"]:
                if key in config and "current" in config[key]:
                    metrics[key] = float(config[key]["current"])
                elif key in config and "initial" in config[key]:
                    metrics[key] = float(config[key]["initial"])
        self._metrics_config[npc] = config
        return metrics

    def extract_metrics(self, response: str) -> Dict[str, any]:
        def extract_tag(text: str, tag: str) -> str:
            start_tag, end_tag = f"<{tag}>", f"</{tag}>"
            start = text.find(start_tag) + len(start_tag)
            end = text.find(end_tag)
            if start == -1 or end == -1:
                return "nochange"
            value = text[start:end].strip()
            return value if value in ["gain", "lose", "nochange"] else "nochange"

        def extract_summary(text: str) -> str:
            start_tag, end_tag = "<summary>", "</summary>"
            start = text.find(start_tag) + len(start_tag)
            end = text.find(end_tag)
            return text[start:end].strip() if start != -1 and end != -1 else ""

        def extract_response(text: str) -> str:
            start_tag, end_tag = "<response>", "</response>"
            start = text.find(start_tag) + len(start_tag)
            end = text.find(end_tag)
            return text[start:end].strip() if start != -1 and end != -1 else ""

        return {
            "summary": extract_summary(response),
            "response": extract_response(response),
            "trust": extract_tag(response, "Trust"),
            "affection": extract_tag(response, "Affection"),
            "respect": extract_tag(response, "Respect")
        }

    def update_metrics_from_response(self, npc: str, response: str) -> None:
        parsed = self.extract_metrics(response)
        current_metrics = self.load_metrics(npc)
        config = self._metrics_config.get(npc, {
            "trust": {"base_gain": 0.5, "base_loss": -1.0, "affection_mod": 0.5, "max": 6, "min": -8, "initial": -4.0, "current": -4.0},
            "affection": {"base_gain": 0.8, "base_loss": -0.8, "max": 8, "min": -10, "initial": 0.0, "current": 0.0},
            "respect": {"base_gain": 0.6, "base_loss": -0.9, "max": 7, "min": -8, "initial": -2.0, "current": -2.0}
        })
        for key in ["trust", "affection", "respect"]:
            action = parsed.get(key, "nochange")
            if action != "nochange":
                if key == "trust":
                    # Normalize affection (-10 to +10) to 0-1
                    affection_norm = (current_metrics["affection"] + 10) / 20
                    mod = config[key]["affection_mod"] * affection_norm
                    gain = config[key]["base_gain"] * (1 + mod)
                    loss = config[key]["base_loss"] * (1 - mod)
                else:
                    gain = config[key]["base_gain"]
                    loss = config[key]["base_loss"]
                delta = gain if action == "gain" else loss
                current_metrics[key] += delta
                current_metrics[key] = max(min(current_metrics[key], config[key]["max"]), config[key]["min"])
                config[key]["current"] = current_metrics[key]
        # Write updated config to npc_metrics.yaml
        metrics_path = self._npc_dir(npc) / "npc_metrics.yaml"
        with metrics_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logger.info("ME: Updated metrics for %s: %s", npc, current_metrics)

    def summarize_history(self, npc: str, history: str) -> None:
        try:
            prompt = self.pe.render(
                "npc_summary",
                npc_name=npc,
                conversation_history=history
            )
            response = self.llm.chat(prompt)
            parsed = self.extract_metrics(response)
            if not parsed.get("summary"):
                logger.warning("ME: Empty summary for %s, history length: %d", npc, len(history))
                return

            summary = parsed["summary"]
            summary_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()
            if summary_hash in self._summary_hashes:
                logger.info("ME: Skipping duplicate summary for %s: %s", npc, summary[:50])
                return
            self._summary_hashes.add(summary_hash)

            metrics_delta = {
                "trust": parsed.get("trust", "nochange"),
                "affection": parsed.get("affection", "nochange"),
                "respect": parsed.get("respect", "nochange")
            }

            # Embed full summary
            embedded = False
            for attempt in range(self.EMBED_RETRY_COUNT):
                try:
                    result = ollama.embeddings(model="mxbai-embed-large", prompt=summary)
                    self.embed.put(summary, f"{npc}_memory", npc=npc)
                    embedded = True
                    break
                except Exception as e:
                    logger.warning("ME: Embed attempt %d failed for %s: %s", attempt + 1, npc, str(e))
                    if attempt < self.EMBED_RETRY_COUNT - 1:
                        time.sleep(self.EMBED_RETRY_DELAY)
            if not embedded:
                logger.error("ME: Failed to embed summary for %s after %d attempts", npc, self.EMBED_RETRY_COUNT)

            # Save to conversation_summary.txt
            summary_path = self._npc_dir(npc) / "conversation_summary.txt"
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(summary + "\n")

            # Save to THE_STORY_SO_FAR.jsonl
            tssf_record = {
                "id": summary_hash,
                "text": summary,
                "hash": summary_hash,
                "npc": npc,
                "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            with self._tssf_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(tssf_record) + "\n")

            # Embed summary chunks
            chunk_prompt = (
                "Divide the text into third-person memory entries. "
                "Each entry describes a single fact, event, or trait. "
                "Separate with <memory-split>.\n\n" + summary
            )
            chunk_response = ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": chunk_prompt}])
            chunks = [c.strip() for c in chunk_response["message"]["content"].split("<memory-split>") if c.strip()]
            emb_path = self._npc_dir(npc) / "conversation_embeddings.jsonl"
            for i, chunk in enumerate(chunks):
                embedded = False
                for attempt in range(self.EMBED_RETRY_COUNT):
                    try:
                        result = ollama.embeddings(model="mxbai-embed-large", prompt=chunk)
                        embedding = result["embedding"]
                        chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        emb_record = {
                            "source": f"summary_{summary_hash}",
                            "source_type": "summary_chunk",
                            "chunk_id": i,
                            "hash": chunk_hash,
                            "content": chunk,
                            "embedding": embedding
                        }
                        with emb_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(emb_record) + "\n")
                        self.embed.put(chunk, f"{npc}_memory", npc=npc)
                        embedded = True
                        break
                    except Exception as e:
                        logger.warning("ME: Embed attempt %d failed for chunk %d of %s: %s", attempt + 1, i, npc, str(e))
                        if attempt < self.EMBED_RETRY_COUNT - 1:
                            time.sleep(self.EMBED_RETRY_DELAY)
                if not embedded:
                    logger.error("ME: Failed to embed chunk %d for %s after %d attempts", i, npc, self.EMBED_RETRY_COUNT)

            # Update relationship metrics from summary
            current_metrics = self.load_metrics(npc)
            config = self._metrics_config.get(npc, {
                "trust": {"base_gain": 0.5, "base_loss": -1.0, "affection_mod": 0.5, "max": 6, "min": -8, "initial": -4.0, "current": -4.0},
                "affection": {"base_gain": 0.8, "base_loss": -0.8, "max": 8, "min": -10, "initial": 0.0, "current": 0.0},
                "respect": {"base_gain": 0.6, "base_loss": -0.9, "max": 7, "min": -8, "initial": -2.0, "current": -2.0}
            })
            for key in ["trust", "affection", "respect"]:
                action = metrics_delta.get(key, "nochange")
                if action != "nochange":
                    if key == "trust":
                        affection_norm = (current_metrics["affection"] + 10) / 20
                        mod = config[key]["affection_mod"] * affection_norm
                        gain = config[key]["base_gain"] * (1 + mod)
                        loss = config[key]["base_loss"] * (1 - mod)
                    else:
                        gain = config[key]["base_gain"]
                        loss = config[key]["base_loss"]
                    delta = gain if action == "gain" else loss
                    current_metrics[key] += delta
                    current_metrics[key] = max(min(current_metrics[key], config[key]["max"]), config[key]["min"])
                    config[key]["current"] = current_metrics[key]
            # Write updated config to npc_metrics.yaml
            metrics_path = self._npc_dir(npc) / "npc_metrics.yaml"
            with metrics_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False)

            # Truncate conversation_history.txt to last KEEP_CHARS and prepend summary
            hist_path = self._history_path(npc)
            if len(history) > self.KEEP_CHARS:
                last_chars = history[-self.KEEP_CHARS:]
                # Start at a complete line
                last_newline = last_chars.rfind("\n")
                if last_newline != -1:
                    last_chars = last_chars[last_newline + 1:]
                new_history = f"Summary: {summary}\n{last_chars}"
                with hist_path.open("w", encoding="utf-8") as fh:
                    fh.write(new_history)
                logger.info("ME: Truncated history for %s to %d chars with summary", npc, len(new_history))

            logger.info("ME: Summarized history for %s: %s", npc, summary)
        except Exception as e:
            logger.error("ME: Summarization failed for %s: %s", npc, str(e))

    def fetch_context(self, npc: str, k_per_bucket: int = 4) -> Dict[str, List[str]]:
        ctx = {
            "character_memories": [],
            "conversation_memories": [],
            "location_memories": [],
            "lore_memories": [],
            "conversation_history": [],
            "static_chunks": [],
            "sheet_chunks": [],
            "memory_chunks": [],
            "metrics": self.load_metrics(npc)
        }
        # Static system instructions
        ctx["static_chunks"] = [
            """
            You are an NPC in the Dragonwarp world, a post-apocalyptic techno-mystical setting.
            Respond in-character, reflecting your personality and the setting's lore.
            Use <safely> to exit if unsafe.
            """
        ]
        # Character sheet
        sheet_path = self._npc_dir(npc) / "character_sheet.txt"
        if sheet_path.exists():
            ctx["sheet_chunks"] = [sheet_path.read_text(encoding="utf-8")]
        else:
            stats_path = self._npc_dir(npc) / "stats.txt"
            personality_path = self._npc_dir(npc) / "personality.txt"
            sheet_data = []
            if stats_path.exists():
                sheet_data.append(stats_path.read_text(encoding="utf-8"))
            if personality_path.exists():
                sheet_data.append(personality_path.read_text(encoding="utf-8"))
            ctx["sheet_chunks"] = sheet_data or ["No character sheet available."]

        # Conversation history
        hist_path = self._history_path(npc)
        if hist_path.exists():
            history = hist_path.read_text(encoding="utf-8")
            ctx["memory_chunks"] = [history]
            ctx["conversation_history"] = history.splitlines()

        # Summaries
        summary_path = self._npc_dir(npc) / "conversation_summary.txt"
        if summary_path.exists():
            ctx["conversation_memories"] = summary_path.read_text(encoding="utf-8").splitlines()

        query = f"{npc} recent conversation"
        if k_per_bucket > 0:
            try:
                ctx["character_memories"] = [hit["text"] for hit in self.embed.search(query, f"{npc}_sheet", k_per_bucket, npc) if "text" in hit]
                ctx["conversation_memories"] += [hit["text"] for hit in self.embed.search(query, f"{npc}_memory", k_per_bucket, npc) if "text" in hit]
                ctx["location_memories"] = [hit["text"] for hit in self.embed.search(query, "location", k_per_bucket) if "text" in hit]
                ctx["lore_memories"] = [hit["text"] for hit in self.embed.search(query, self._static_domain, k_per_bucket) if "text" in hit]
            except Exception as e:
                logger.error("ME: Failed to fetch embeddings for %s: %s", npc, str(e))
        logger.debug(
            "ME: fetch_context %s → char:%d conv:%d loc:%d lore:%d hist:%d static:%d sheet:%d mem:%d",
            npc, len(ctx["character_memories"]), len(ctx["conversation_memories"]),
            len(ctx["location_memories"]), len(ctx["lore_memories"]), len(ctx["conversation_history"]),
            len(ctx["static_chunks"]), len(ctx["sheet_chunks"]), len(ctx["memory_chunks"])
        )
        return ctx