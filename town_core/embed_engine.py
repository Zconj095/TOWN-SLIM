from __future__ import annotations
import hashlib, json, logging, os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
import ollama
from town_core.llm_engine import LLMEngine

logger = logging.getLogger("EE")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s [%(asctime)s]"))
    logger.addHandler(h)

class _FlatJSONLStore:
    def __init__(self, storage_dir: str | Path = "npc_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_for(self, domain: str, npc: str = None) -> List[Path]:
        if domain == "static_lore":
            return [self.storage_dir / "dragonwarp_lore.jsonl"]
        elif domain == "location":
            return [self.storage_dir / "current_location.jsonl"]
        elif domain.endswith("_sheet"):
            npc = domain[:-6]
            return [self.storage_dir / npc / "embeddings.jsonl"]
        elif domain.endswith("_memory"):
            npc = domain[:-7]
            paths = [self.storage_dir / npc / "conversation_embeddings.jsonl"]
            # Check both npc folder and npc_data for THE_STORY_SO_FAR.jsonl
            story_paths = [
                self.storage_dir / npc / "THE_STORY_SO_FAR.jsonl",
                self.storage_dir / "THE_STORY_SO_FAR.jsonl"
            ]
            for story_path in story_paths:
                if story_path.exists():
                    paths.append(story_path)
            return paths
        return [self.storage_dir / f"{domain}.jsonl"]

    def _load_vectors(self, domain: str, npc: str = None) -> List[Dict[str, Any]]:
        paths = self._file_for(domain, npc)
        entries = []
        for path in paths:
            if not path.exists():
                logger.debug("EE: No file at %s", path)
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip())
                        text = obj.get("text", obj.get("content", ""))
                        if not text:
                            logger.debug("EE: Skipping empty text in %s", path)
                            continue
                        if "embedding" not in obj:
                            logger.warning("EE: Missing embedding in %s", path)
                            continue
                        # Generate id if missing
                        obj["id"] = obj.get("id", hashlib.sha256(text.encode("utf-8")).hexdigest())
                        entries.append(obj)
                    except json.JSONDecodeError:
                        logger.debug("EE: Invalid JSON line in %s", path)
        return entries

    def add(self, record: Dict[str, Any]) -> None:
        paths = self._file_for(record["domain"], record.get("npc"))
        path = paths[0]  # Write to first path (e.g., conversation_embeddings.jsonl for _memory)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def query(self, vector: List[float], domain: str, k: int, npc: str = None) -> List[Dict[str, Any]]:
        entries = self._load_vectors(domain, npc)
        if not entries:
            return []
        # Build FAISS index
        vectors = np.array([e["embedding"] for e in entries], dtype=np.float32)
        idx = faiss.IndexFlatL2(vectors.shape[1])
        idx.add(vectors)
        # Query
        v = np.array(vector, dtype=np.float32).reshape(1, -1)
        _, I = idx.search(v, k)
        return [{"id": entries[i]["id"], "text": entries[i]["text"], "score": 1.0} for i in I[0] if i < len(entries)]

class EmbedEngine:
    def __init__(self, storage_dir: str | Path = "npc_data"):
        self.store = _FlatJSONLStore(storage_dir)
        self.llm = LLMEngine()

    def put(self, text: str, domain: str, npc: str = None) -> str:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        paths = self.store._file_for(domain, npc)
        for path in paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            obj = json.loads(line.strip())
                            obj_text = obj.get("text", obj.get("content", ""))
                            if hashlib.sha256(obj_text.encode("utf-8")).hexdigest() == sha:
                                logger.debug("EE: Duplicate text skipped (%s)", sha[:8])
                                return sha
                        except json.JSONDecodeError:
                            logger.debug("EE: Invalid JSON line in %s", path)
        vector = ollama.embeddings(model="mxbai-embed-large", prompt=text)["embedding"]
        record = {
            "id": sha,
            "domain": domain,
            "text": text,
            "embedding": vector,
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z"
        }
        if npc:
            record["npc"] = npc
        self.store.add(record)
        logger.info("EE: Added record %s (%s)", sha[:8], domain)
        return sha

    def search(self, query: str, domain: str, k: int = 5, npc: str = None) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        vector = ollama.embeddings(model="mxbai-embed-large", prompt=query)["embedding"]
        results = self.store.query(vector, domain, k, npc)
        logger.debug("EE: search '%s' -> %d hits", query[:30], len(results))
        return results

if __name__ == "__main__":
    import sys
    usage = "python embed_engine.py \"text to add/search\" <domain> [--search k] [--npc name]"
    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)
    txt = sys.argv[1]
    dom = sys.argv[2]
    npc = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--npc" else None
    k_val = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--search" else 5
    ee = EmbedEngine()
    if "--search" in sys.argv:
        res = ee.search(txt, dom, k=k_val, npc=npc)
        for r in res:
            print(f"{r['score']:.3f}  {r['text'][:80]}")
    else:
        rid = ee.put(txt, dom, npc=npc)
        print("Added id:", rid)