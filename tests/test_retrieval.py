"""
Unit tests for Hybrid Retrieval Engine (BM25 + Semantic + RRF).
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.retriever import (
    get_relevant_memories, semantic_search,
    keyword_search, reciprocal_rank_fusion, get_all_docs
)


@pytest.fixture(autouse=True)
def seed():
    from src.storage.vector_store import clear_all_memories, save_memories
    clear_all_memories()
    save_memories([
        {"memory_type": "semantic",   "sub_type": "fact",       "key": "name",        "value": "Ayush",             "confidence": 1.0,  "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "preference", "key": "call_time",   "value": "after 11 AM",       "confidence": 0.95, "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "constraint", "key": "diet",        "value": "vegetarian",        "confidence": 1.0,  "source_turn": 2},
        {"memory_type": "episodic",   "sub_type": "event",      "key": "recent_trip", "value": "went to Delhi",     "confidence": 0.9,  "source_turn": 3},
        {"memory_type": "procedural", "sub_type": "style",      "key": "style",       "value": "use bullet points", "confidence": 1.0,  "source_turn": 4},
    ])
    yield


class TestHybridRetrieval:

    def test_returns_dict(self):
        result = get_relevant_memories("call me tomorrow")
        assert isinstance(result, dict)
        assert "query" in result
        assert "memories" in result
        assert "count" in result

    def test_semantic_call_time(self):
        result = get_relevant_memories("Can you call me tomorrow?", top_k=3)
        values = [m["value"].lower() for m in result["memories"]]
        assert any("11 am" in v for v in values)

    def test_keyword_exact_match(self):
        result = get_relevant_memories("11 AM", top_k=3)
        values = [m["value"].lower() for m in result["memories"]]
        assert any("11 am" in v for v in values)

    def test_diet_retrieval(self):
        result = get_relevant_memories("I am vegetarian what should I eat?", top_k=3)
        values = [m["value"].lower() for m in result["memories"]]
        assert any("vegetarian" in v for v in values)

    def test_episodic_retrieval(self):
        result = get_relevant_memories("Did I go anywhere recently?", top_k=3)
        types = [m["memory_type"] for m in result["memories"]]
        assert "episodic" in types

    def test_procedural_retrieval(self):
        result = get_relevant_memories("use bullet points to respond", top_k=3)
        types = [m["memory_type"] for m in result["memories"]]
        assert "procedural" in types

    def test_rrf_scores_present(self):
        result = get_relevant_memories("Ayush", top_k=3)
        for mem in result["memories"]:
            assert "score" in mem
            assert mem["score"] > 0

    def test_top_k_respected(self):
        result = get_relevant_memories("anything", top_k=2)
        assert result["count"] <= 2