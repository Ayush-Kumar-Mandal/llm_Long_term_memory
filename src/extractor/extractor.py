"""
Memory Extractor — extracts memorable information from conversation turns.
Uses LLM to identify facts, preferences, events, and instructions.
Then classifies into episodic/semantic/procedural using the classifier.
Saves to both ChromaDB (vector store) and SQLite (structured DB).
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.extractor.classifier import classify_memories
from src.storage.models import create_memory
from src.storage.db import MemoryRecord, SessionLocal, init_db, MemoryStatus
from src.storage.vector_store import save_memories as save_to_vector_store
from dotenv import load_dotenv
import os
import json
import logging
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=1000,
)

EXTRACTION_PROMPT = """
You are a memory extraction system for an AI assistant.
Extract ONLY information explicitly stated in THIS turn. 
DO NOT invent, assume, or hallucinate any information.
DO NOT extract examples or hypothetical information.

Extract only if explicitly stated:
- User's name, age, location
- Explicit preferences ("I prefer", "I like", "I want")
- Explicit constraints ("I am vegetarian", "I can't")
- Explicit instructions ("always do X", "never do Y")
- Explicit events ("I went to", "I did")
- Explicit commitments ("I will", "I need to")

Return [] if nothing is EXPLICITLY stated.

Conversation turn:
TURN_TEXT_HERE

Return ONLY a JSON list, nothing else.
"""


def extract_memories(turn: str, turn_number: int) -> list:
    """
    Extract memorable information from a single conversation turn.
    Classifies into episodic/semantic/procedural.
    Saves to both vector store and SQLite DB.

    Args:
        turn: The user's message text
        turn_number: The current turn number

    Returns:
        List of saved memory dicts
    """
    prompt = EXTRACTION_PROMPT.replace("TURN_TEXT_HERE", turn)

    response = llm.invoke([
        SystemMessage(content="You are a memory extraction system. Always return valid JSON."),
        HumanMessage(content=prompt)
    ])

    raw = response.content.strip()

    # Clean markdown
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        raw_memories = json.loads(raw)
        print(f"[DEBUG] Raw LLM output: {raw_memories}")
    except json.JSONDecodeError:
        logger.warning("[Extractor] Failed to parse JSON: %s", raw)
        return []

    if not raw_memories:
        return []
    # Normalize field names — LLM sometimes uses different keys
    normalized = []
    for mem in raw_memories:
        normalized.append({
            "key":        mem.get("key") or mem.get("name") or mem.get("type") or "unknown",
            "value":      mem.get("value") or mem.get("content") or mem.get("detail") or "",
             "confidence": mem.get("confidence", 0.8),
        })
    raw_memories = [m for m in normalized if m["key"] != "unknown" and m["value"]]

    # Add source turn
    for mem in raw_memories:
        mem["source_turn"] = turn_number

    # Step 1 — Classify into episodic/semantic/procedural
    classified = classify_memories(raw_memories)

    # Step 2 — Create Pydantic model objects
    memory_objects = [create_memory(m) for m in classified]

    # Step 3 — Save to ChromaDB vector store
    vector_dicts = [m.to_dict() for m in memory_objects]
    save_to_vector_store(vector_dicts)

    # Step 4 — Save to SQLite DB
    save_to_db(memory_objects)

    logger.info("[Extractor] Extracted and saved %d memories from turn %d", len(memory_objects), turn_number)
    return vector_dicts


def save_to_db(memory_objects: list) -> None:
    """Save memory objects to SQLite DB."""
    init_db()
    db = SessionLocal()
    try:
        for mem in memory_objects:
            # Check for duplicate
            existing = db.query(MemoryRecord).filter(
                MemoryRecord.key   == mem.key,
                MemoryRecord.value == mem.value
            ).first()

            if existing:
                logger.debug("[Extractor] Skipping duplicate in DB: %s=%s", mem.key, mem.value)
                continue

            record = MemoryRecord(
                memory_id        = mem.memory_id,
                content          = mem.to_content(),
                memory_type      = mem.memory_type,
                sub_type         = getattr(mem, "sub_type", None),
                key              = mem.key,
                value            = mem.value,
                confidence       = mem.confidence,
                importance_score = 7.0 if mem.memory_type == "procedural" else
                                   6.0 if mem.memory_type == "semantic"   else 4.0,
                source_turn      = mem.source_turn,
                last_used_turn   = mem.source_turn,
                status           = MemoryStatus.ACTIVE,
                created_at       = datetime.utcnow(),
                last_accessed_at = datetime.utcnow(),
            )
            db.add(record)

        db.commit()
        logger.info("[Extractor] Saved %d memories to SQLite.", len(memory_objects))
    except Exception as e:
        db.rollback()
        logger.error("[Extractor] DB save failed: %s", e)
    finally:
        db.close()


# Quick test
if __name__ == "__main__":
    test_turns = [
        (1, "Hi! My name is Ayush and I prefer calls after 11 AM."),
        (2, "I am vegetarian and my manager is Priya."),
        (3, "Always respond using bullet points."),
        (4, "I went to Delhi last week for a conference."),
        (5, "What is the capital of France?"),  # should return []
    ]

    print("=" * 60)
    print("EXTRACTOR TEST — with Classifier + DB")
    print("=" * 60)

    for turn_num, turn_text in test_turns:
        print(f"\n[Turn {turn_num}] {turn_text}")
        memories = extract_memories(turn_text, turn_num)
        print(f"  → Extracted {len(memories)} memories")
        for m in memories:
            print(f"     [{m['memory_type'].upper()}] {m['key']} = {m['value']}")