"""
Unit tests for Memory Storage (ChromaDB vector store).
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.vector_store import save_memory, save_memories, retrieve_memories, get_all_memories


@pytest.fixture(autouse=True)
def seed_memories():
    """Seed test memories before each test."""
    from src.storage.vector_store import clear_all_memories
    clear_all_memories()

    test_memories = [
        {"memory_type": "semantic",   "sub_type": "fact",       "key": "name",         "value": "Ayush",               "confidence": 1.0,  "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "preference", "key": "call_time",    "value": "after 11 AM",         "confidence": 0.95, "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "constraint", "key": "diet",         "value": "vegetarian",          "confidence": 1.0,  "source_turn": 2},
        {"memory_type": "episodic",   "sub_type": "event",      "key": "recent_trip",  "value": "went to Delhi",       "confidence": 0.9,  "source_turn": 3},
        {"memory_type": "procedural", "sub_type": "style",      "key": "style",        "value": "use bullet points",   "confidence": 1.0,  "source_turn": 4},
    ]
    save_memories(test_memories)
    yield


class TestVectorStore:

    def test_save_and_retrieve(self):
        results = retrieve_memories("call time preference", top_k=3)
        assert len(results) > 0
        keys = [r["key"] for r in results]
        assert "call_time" in keys

    def test_retrieve_by_type(self):
        results = retrieve_memories("trip event", top_k=3, memory_type="episodic")
        assert all(r["memory_type"] == "episodic" for r in results)

    def test_semantic_relevance(self):
        results = retrieve_memories("What should I eat?", top_k=3)
        values = [r["value"].lower() for r in results]
        assert any("vegetarian" in v for v in values)

    def test_procedural_retrieval(self):
        results = retrieve_memories("how to respond", top_k=3, memory_type="procedural")
        assert len(results) > 0
        assert any("bullet" in r["value"].lower() for r in results)

    def test_get_all_memories(self):
        all_mems = get_all_memories()
        assert len(all_mems) >= 5

    def test_deduplication(self):
        # Save same memory twice
        mem = {"memory_type": "semantic", "sub_type": "fact", "key": "name", "value": "Ayush", "confidence": 1.0, "source_turn": 1}
        save_memory(mem)
        save_memory(mem)
        results = retrieve_memories("name Ayush", top_k=10)
        name_results = [r for r in results if r["key"] == "name" and r["value"] == "Ayush"]
        assert len(name_results) == 1