"""
seed_memory.py — pre-loads a rich set of memories for testing.
Useful for quickly setting up a realistic memory state.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.vector_store import save_memories, clear_all_memories
from src.storage.db import init_db, SessionLocal, MemoryRecord, MemoryStatus
from datetime import datetime, timedelta

SEED_MEMORIES = [
    # Semantic — facts
    {"memory_type": "semantic", "sub_type": "fact",       "key": "name",           "value": "Ayush",                        "confidence": 1.0,  "importance_score": 9.0, "source_turn": 1},
    {"memory_type": "semantic", "sub_type": "fact",       "key": "location",       "value": "Roorkee",                      "confidence": 1.0,  "importance_score": 8.0, "source_turn": 1},
    {"memory_type": "semantic", "sub_type": "fact",       "key": "occupation",     "value": "student at IIT Roorkee",       "confidence": 0.95, "importance_score": 8.0, "source_turn": 2},

    # Semantic — preferences
    {"memory_type": "semantic", "sub_type": "preference", "key": "call_time",      "value": "after 11 AM",                  "confidence": 0.95, "importance_score": 7.0, "source_turn": 1},
    {"memory_type": "semantic", "sub_type": "preference", "key": "language",       "value": "Kannada",                      "confidence": 0.9,  "importance_score": 7.0, "source_turn": 3},

    # Semantic — constraints
    {"memory_type": "semantic", "sub_type": "constraint", "key": "diet",           "value": "vegetarian",                   "confidence": 1.0,  "importance_score": 8.0, "source_turn": 2},

    # Semantic — entities
    {"memory_type": "semantic", "sub_type": "entity",     "key": "manager",        "value": "Priya",                        "confidence": 1.0,  "importance_score": 7.0, "source_turn": 4},

    # Semantic — commitments
    {"memory_type": "semantic", "sub_type": "commitment", "key": "report_due",     "value": "Friday",                       "confidence": 0.9,  "importance_score": 7.0, "source_turn": 5},

    # Episodic — events
    {"memory_type": "episodic", "sub_type": "event",      "key": "recent_trip",    "value": "went to Delhi last week",      "confidence": 0.9,  "importance_score": 5.0, "source_turn": 10},
    {"memory_type": "episodic", "sub_type": "experience", "key": "hackathon",      "value": "participating in IITG hackathon", "confidence": 1.0, "importance_score": 6.0, "source_turn": 15},

    # Procedural — instructions
    {"memory_type": "procedural", "sub_type": "style",       "key": "response_style", "value": "always use bullet points",  "confidence": 1.0,  "importance_score": 8.0, "source_turn": 5},
    {"memory_type": "procedural", "sub_type": "instruction",  "key": "tone",           "value": "formal and concise",        "confidence": 0.9,  "importance_score": 8.0, "source_turn": 6},
]


def seed():
    print("=" * 60)
    print("SEEDING MEMORY STORE")
    print("=" * 60)

    # Clear existing
    clear_all_memories()
    print("\n[1] Cleared existing memories.")

    # Save to ChromaDB
    save_memories(SEED_MEMORIES)
    print(f"\n[2] Saved {len(SEED_MEMORIES)} memories to ChromaDB.")

    # Save to SQLite
    init_db()
    db = SessionLocal()
    try:
        for mem in SEED_MEMORIES:
            existing = db.query(MemoryRecord).filter(
                MemoryRecord.key   == mem["key"],
                MemoryRecord.value == mem["value"]
            ).first()
            if existing:
                continue

            record = MemoryRecord(
                memory_id        = f"seed_{mem['key']}",
                content          = f"{mem['memory_type']}: {mem['key']} is {mem['value']}",
                memory_type      = mem["memory_type"],
                sub_type         = mem.get("sub_type"),
                key              = mem["key"],
                value            = mem["value"],
                confidence       = mem["confidence"],
                importance_score = mem.get("importance_score", 5.0),
                source_turn      = mem["source_turn"],
                last_used_turn   = mem["source_turn"],
                status           = MemoryStatus.ACTIVE,
                created_at       = datetime.utcnow() - timedelta(days=1),
                last_accessed_at = datetime.utcnow(),
            )
            db.add(record)
        db.commit()
        print(f"[3] Saved {len(SEED_MEMORIES)} memories to SQLite.")
    finally:
        db.close()

    print("\n✅ Memory store seeded successfully!")
    print("\nMemories by type:")
    by_type = {}
    for m in SEED_MEMORIES:
        by_type[m["memory_type"]] = by_type.get(m["memory_type"], 0) + 1
    for t, count in by_type.items():
        print(f"  {t:12} : {count}")


if __name__ == "__main__":
    seed()