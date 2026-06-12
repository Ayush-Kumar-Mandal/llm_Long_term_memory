"""
MemoryCompressor — merges semantically similar memories into summaries.
Preserves links to source memories.
"""

import logging
import json
from typing import List
from sqlalchemy.orm import Session
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import uuid
import numpy as np

from src.storage.db import MemoryRecord, MemoryStatus, SessionLocal, init_db
from src.storage.schemas import CompressionResult

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=500,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

COMPRESS_PROMPT = """
You are a memory compression system.
Given these related memories, create a single concise summary that preserves all key information.

Memories:
{memories}

Return a single JSON object:
{{
  "key": "compressed_summary_key",
  "value": "concise summary preserving all facts"
}}
Return ONLY JSON, nothing else.
"""

SIMILARITY_THRESHOLD = 0.85


class MemoryCompressor:
    """
    Merges semantically similar memories into compressed summaries.
    """

    def __init__(self):
        init_db()
        logger.info("[Compressor] Initialized.")

    def _cosine_similarity(self, a: list, b: list) -> float:
        """Compute cosine similarity between two embeddings."""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def find_similar_groups(self, memories: List[MemoryRecord]) -> List[List[MemoryRecord]]:
        """
        Group memories by semantic similarity.
        Returns list of groups, each group to be compressed into one memory.
        """
        if not memories:
            return []

        # Get embeddings for all memories
        contents = [m.content for m in memories]
        embeds   = embeddings.embed_documents(contents)

        visited = set()
        groups  = []

        for i, mem_i in enumerate(memories):
            if mem_i.memory_id in visited:
                continue
            group = [mem_i]
            visited.add(mem_i.memory_id)

            for j, mem_j in enumerate(memories):
                if i == j or mem_j.memory_id in visited:
                    continue
                sim = self._cosine_similarity(embeds[i], embeds[j])
                if sim >= SIMILARITY_THRESHOLD:
                    group.append(mem_j)
                    visited.add(mem_j.memory_id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def compress_group(self, db: Session, group: List[MemoryRecord]) -> CompressionResult:
        """
        Compress a group of similar memories into one.
        """
        memory_texts = "\n".join([
            f"- [{m.memory_type}] {m.key}: {m.value}"
            for m in group
        ])

        prompt = COMPRESS_PROMPT.replace("{memories}", memory_texts)
        response = llm.invoke([
            SystemMessage(content="You are a memory compression system. Return valid JSON only."),
            HumanMessage(content=prompt)
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"key": "compressed", "value": memory_texts}

        # Create new compressed memory
        new_id      = f"mem_{uuid.uuid4().hex[:8]}"
        source_ids  = [m.memory_id for m in group]
        best_type   = max(set([m.memory_type for m in group]), key=lambda t: [m.memory_type for m in group].count(t))
        best_importance = max(m.importance_score for m in group)

        new_memory = MemoryRecord(
            memory_id        = new_id,
            content          = f"{best_type}: {result['key']} is {result['value']}",
            memory_type      = best_type,
            key              = result["key"],
            value            = result["value"],
            importance_score = best_importance,
            confidence       = sum(m.confidence for m in group) / len(group),
            access_count     = sum(m.access_count for m in group),
            source_turn      = min(m.source_turn for m in group),
            is_compressed    = True,
            compressed_from  = json.dumps(source_ids),
            status           = MemoryStatus.ACTIVE
        )

        db.add(new_memory)

        # Mark source memories as compressed
        for m in group:
            m.status = MemoryStatus.COMPRESSED

        db.commit()
        logger.info("[Compressor] Compressed %d memories → %s", len(group), new_id)

        return CompressionResult(
            new_memory_id       = new_id,
            source_memory_ids   = source_ids,
            compressed_content  = new_memory.content,
            memories_reduced    = len(group) - 1
        )

    def run_compression(self, db: Session) -> List[CompressionResult]:
        """
        Find and compress all similar memory groups.
        """
        memories = db.query(MemoryRecord).filter(
            MemoryRecord.status == MemoryStatus.COMPRESSED
        ).all()

        if not memories:
            logger.info("[Compressor] No memories to compress.")
            return []

        groups  = self.find_similar_groups(memories)
        results = []

        for group in groups:
            result = self.compress_group(db, group)
            results.append(result)

        logger.info("[Compressor] Compressed %d groups.", len(results))
        return results


# Quick test
if __name__ == "__main__":
    from src.storage.db import init_db, SessionLocal, MemoryRecord, MemoryStatus
    from datetime import datetime

    init_db()
    db = SessionLocal()

    # Seed similar memories
    similar = [
        MemoryRecord(memory_id="mem_c01", content="semantic: call_time is after 11 AM",       memory_type="semantic", key="call_time",  value="after 11 AM",      importance_score=7.0, confidence=0.9, access_count=5, source_turn=1, status=MemoryStatus.COMPRESSED),
        MemoryRecord(memory_id="mem_c02", content="semantic: preferred_call is post 11 AM",   memory_type="semantic", key="call_pref",  value="post 11 AM",       importance_score=6.0, confidence=0.8, access_count=3, source_turn=2, status=MemoryStatus.COMPRESSED),
        MemoryRecord(memory_id="mem_c03", content="semantic: morning_availability is after 11",memory_type="semantic", key="availability","value":"after 11",      importance_score=5.0, confidence=0.7, access_count=2, source_turn=3, status=MemoryStatus.COMPRESSED),
    ]

    for m in similar:
        existing = db.query(MemoryRecord).filter(MemoryRecord.memory_id == m.memory_id).first()
        if not existing:
            db.add(m)
    db.commit()

    compressor = MemoryCompressor()
    results    = compressor.run_compression(db)

    print("\n" + "="*60)
    print("COMPRESSION RESULTS")
    print("="*60)
    for r in results:
        print(f"  New memory   : {r.new_memory_id}")
        print(f"  Sources      : {r.source_memory_ids}")
        print(f"  Content      : {r.compressed_content}")
        print(f"  Reduced by   : {r.memories_reduced} memories")

    db.close()