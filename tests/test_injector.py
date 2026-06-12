"""
Unit tests for Context Injector and Context Builder.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.injector.context_builder import ContextBuilder
from src.injector.injector import process_turn


@pytest.fixture(autouse=True)
def seed():
    from src.storage.vector_store import clear_all_memories, save_memories
    clear_all_memories()
    save_memories([
        {"memory_type": "semantic",   "sub_type": "fact",       "key": "name",       "value": "Ayush",           "confidence": 1.0,  "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "preference", "key": "call_time",  "value": "after 11 AM",     "confidence": 0.95, "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "constraint", "key": "diet",       "value": "vegetarian",      "confidence": 1.0,  "source_turn": 2},
        {"memory_type": "procedural", "sub_type": "style",      "key": "style",      "value": "bullet points",   "confidence": 1.0,  "source_turn": 3},
    ])
    yield


class TestContextBuilder:

    def test_build_returns_dict(self):
        builder = ContextBuilder()
        result  = builder.build("Can you call me?")
        assert "system_prompt" in result
        assert "memory_block" in result
        assert "active_memories" in result

    def test_memory_block_not_empty(self):
        builder = ContextBuilder()
        result  = builder.build("Can you call me?")
        assert result["memory_block"] != ""

    def test_system_prompt_contains_memories(self):
        builder = ContextBuilder()
        result  = builder.build("Can you call me?")
        assert "11 AM" in result["system_prompt"] or "call" in result["system_prompt"].lower()

    def test_procedural_in_prompt(self):
        builder = ContextBuilder()
        result  = builder.build("How should you respond?")
        assert "bullet" in result["system_prompt"].lower()

    def test_archive_fallback(self):
        builder = ContextBuilder()
        # Query something completely unrelated — should not crash
        result  = builder.build("xyz123 random nonsense query")
        assert "system_prompt" in result


class TestInjector:

    def test_process_turn_returns_dict(self):
        history = []
        result  = process_turn("Hello!", 1, history)
        assert "response" in result
        assert "active_memories" in result
        assert "new_memories_saved" in result

    def test_response_not_empty(self):
        history = []
        result  = process_turn("What is 2+2?", 1, history)
        assert len(result["response"]) > 0

    def test_history_updated(self):
        history = []
        process_turn("My name is Ayush.", 1, history)
        assert len(history) == 2  # user + assistant messages

    def test_memory_injection_in_response(self):
        history = []
        result  = process_turn("Can you call me tomorrow?", 1, history)
        # Should mention 11 AM from seeded memories
        assert "11" in result["response"] or "call" in result["response"].lower()