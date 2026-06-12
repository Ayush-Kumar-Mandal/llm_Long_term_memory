"""
ContextBuilder — builds memory-augmented system prompts.
Retrieves from all 3 memory collections and formats
into a structured prompt block for LLM injection.
"""

import logging
from typing import Optional
from src.retrieval.retriever import get_relevant_memories, format_memories_for_prompt
from src.storage.db import MemoryRecord, SessionLocal, MemoryStatus
from src.memory.decay_engine import MemoryDecayEngine
from src.storage.schemas import DecayConfig

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """You are a helpful AI assistant with long-term memory.
You remember important things about the user from past conversations.
Use remembered information naturally without saying "I remember" or "you told me".
Be concise, helpful, and personalized."""


class ContextBuilder:
    """
    Builds memory-augmented system prompts for LLM inference.

    Retrieves from:
    - Semantic memory  → facts, preferences, entities
    - Episodic memory  → past events and experiences
    - Procedural memory → behavior rules and instructions

    Applies decay reinforcement on retrieval.
    Falls back to archive if primary retrieval returns nothing.
    """

    def __init__(self, config: DecayConfig = None, top_k: int = 5):
        self.config  = config or DecayConfig()
        self.top_k   = top_k
        self.decay_engine = MemoryDecayEngine(config=self.config)
        logger.info("[ContextBuilder] Initialized.")

    def build(self, query: str, turn_number: int = 0) -> dict:
        """
        Build a memory-augmented system prompt for a given query.

        Args:
            query: Current user message
            turn_number: Current turn number

        Returns:
            dict with:
                - system_prompt: full prompt with memories injected
                - memory_block: just the memory section
                - active_memories: list of retrieved memory dicts
                - from_archive: whether archive was used
        """
        # Step 1 — Retrieve from all 3 collections
        memory_data  = get_relevant_memories(query, top_k=self.top_k)
        from_archive = False

        # Step 2 — Fallback to archive if nothing found
        if not memory_data["memories"]:
            logger.info("[ContextBuilder] Primary retrieval empty — searching archive.")
            memory_data  = self._search_archive(query)
            from_archive = True

        # Step 3 — Reinforce retrieved memories in SQLite
        if memory_data["memories"]:
            self._reinforce_memories(memory_data["memories"], turn_number)

        # Step 4 — Format memory block
        memory_block = format_memories_for_prompt(memory_data)

        # Step 5 — Build full system prompt
        if memory_block:
            system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{memory_block}"
        else:
            system_prompt = BASE_SYSTEM_PROMPT

        return {
            "system_prompt":    system_prompt,
            "memory_block":     memory_block,
            "active_memories":  memory_data["memories"],
            "from_archive":     from_archive,
            "memory_count":     memory_data["count"]
        }

    def _search_archive(self, query: str) -> dict:
        """Search cold storage as fallback."""
        from src.memory.archive_manager import ArchiveManager
        db      = SessionLocal()
        manager = ArchiveManager()

        # Extract keywords from query for archive search
        keywords = [w for w in query.lower().split() if len(w) > 3]
        results  = []

        for kw in keywords[:3]:
            archived = manager.search_archive(db, query_key=kw)
            for a in archived:
                results.append({
                    "content":     a.content,
                    "memory_type": a.memory_type,
                    "key":         a.key,
                    "value":       a.value,
                    "source_turn": a.source_turn,
                    "score":       0.1,  # low score since from archive
                    "sub_type":    "archived"
                })

        db.close()
        return {"query": query, "memories": results, "count": len(results)}

    def _reinforce_memories(self, memories: list, turn_number: int) -> None:
        """Reinforce retrieved memories in SQLite decay tracking."""
        db = SessionLocal()
        try:
            for mem in memories:
                memory_id = mem.get("memory_id")
                if not memory_id:
                    continue
                record = db.query(MemoryRecord).filter(
                    MemoryRecord.memory_id == memory_id
                ).first()
                if record:
                    self.decay_engine.update_on_retrieval(db, memory_id)
                    record.last_used_turn = turn_number
            db.commit()
        except Exception as e:
            logger.warning("[ContextBuilder] Reinforce failed: %s", e)
        finally:
            db.close()


# Quick test
if __name__ == "__main__":
    print("Starting context builder test...")
    builder = ContextBuilder(top_k=5)

    test_queries = [
        "Can you call me tomorrow?",
        "What should I eat for lunch?",
        "How should you respond to me?",
        "Did I go anywhere recently?",
        "What is my name?",
    ]

    print("=" * 60)
    print("CONTEXT BUILDER TEST")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = builder.build(query, turn_number=10)
        print(f"Memories found : {result['memory_count']}")
        print(f"From archive   : {result['from_archive']}")
        print(f"Memory block:\n{result['memory_block']}")
        print("-" * 40)