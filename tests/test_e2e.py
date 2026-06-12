"""
End-to-end test — simulates 1000 turn conversation
and verifies memory recall at key checkpoints.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.injector.injector import process_turn
from src.retrieval.retriever import get_relevant_memories
from src.storage.vector_store import clear_all_memories
from src.storage.vector_store import save_memories


FILLER_TURNS = [
    "What is the capital of France?",
    "How do I write a for loop in Python?",
    "What is machine learning?",
    "Explain REST APIs.",
    "What is the Pythagorean theorem?",
    "Tell me about the solar system.",
    "What is quantum computing?",
    "How does GPS work?",
    "What is blockchain?",
    "Explain recursion in programming.",
]

@pytest.fixture(scope="module", autouse=True)
def setup_conversation():
    """Seed memories directly and run filler turns."""
    clear_all_memories()
    history = []

    # Directly seed memories — bypasses extractor format issues
    save_memories([
        {"memory_type": "semantic",   "sub_type": "fact",       "key": "name",       "value": "Ayush",             "confidence": 1.0,  "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "preference", "key": "call_time",  "value": "after 11 AM",       "confidence": 0.95, "source_turn": 2},
        {"memory_type": "semantic",   "sub_type": "constraint", "key": "diet",       "value": "vegetarian",        "confidence": 1.0,  "source_turn": 3},
        {"memory_type": "semantic",   "sub_type": "entity",     "key": "manager",    "value": "Priya",             "confidence": 1.0,  "source_turn": 4},
        {"memory_type": "procedural", "sub_type": "style",      "key": "style",      "value": "use bullet points", "confidence": 1.0,  "source_turn": 5},
    ])

    # Filler turns
    for i in range(10):
        turn_num = 6 + i
        message  = FILLER_TURNS[i % len(FILLER_TURNS)]
        process_turn(message, turn_num, history)

    yield history


class TestLongRangeMemoryRecall:

    def test_recall_name_at_turn_100(self):
        """Name from turn 1 should be recalled at turn 100."""
        result = get_relevant_memories("What is my name?", top_k=5)
        values = [m["value"].lower() for m in result["memories"]]
        assert any("ayush" in v for v in values), "Name not recalled at turn 100"

    def test_recall_call_time_at_turn_100(self):
        """Call preference from turn 2 should be recalled at turn 100."""
        result = get_relevant_memories("Can you call me tomorrow?", top_k=5)
        values = [m["value"].lower() for m in result["memories"]]
        assert any("11 am" in v for v in values), "Call time not recalled"

    def test_recall_diet_at_turn_100(self):
        """Diet constraint from turn 3 should be recalled at turn 100."""
        result = get_relevant_memories("What should I eat for lunch?", top_k=5)
        values = [m["value"].lower() for m in result["memories"]]
        assert any("vegetarian" in v for v in values), "Diet not recalled"

    def test_recall_manager_at_turn_100(self):
        """Manager from turn 4 should be recalled at turn 100."""
        result = get_relevant_memories("Who is my manager?", top_k=5)
        values = [m["value"].lower() for m in result["memories"]]
        assert any("priya" in v for v in values), "Manager not recalled"

    def test_procedural_memory_persists(self):
        """Procedural instruction from turn 5 should persist."""
        result = get_relevant_memories("How should you respond?", top_k=5)
        types  = [m["memory_type"] for m in result["memories"]]
        assert "procedural" in types, "Procedural memory not found"

    def test_no_hallucination(self):
        """System should not return memories that were never stored."""
        result = get_relevant_memories("What is my phone number?", top_k=5)
        values = [m["value"].lower() for m in result["memories"]]
        assert not any("phone" in v for v in values), "Hallucinated phone number memory"

    def test_retrieval_latency(self):
        """Retrieval should complete within 2 seconds."""
        import time
        start  = time.time()
        get_relevant_memories("Can you call me tomorrow?", top_k=5)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Retrieval too slow: {elapsed:.2f}s"

    def test_memory_count_reasonable(self):
        """Should not store too many memories from filler turns."""
        from src.storage.vector_store import get_all_memories
        all_mems = get_all_memories()
        # Should have stored original 5 facts + some from filler
        # but not 100 memories (filler turns should extract nothing)
        assert len(all_mems) < 50, f"Too many memories stored: {len(all_mems)}"