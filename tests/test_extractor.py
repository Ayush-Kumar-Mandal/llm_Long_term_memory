"""
Unit tests for Memory Extractor and Classifier.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extractor.classifier import classify_memory, classify_memories


class TestClassifier:

    def test_semantic_fact(self):
        result = classify_memory("name", "Ayush")
        assert result["memory_type"] == "semantic"
        assert result["importance_score"] >= 6.0

    def test_semantic_preference(self):
        result = classify_memory("call_time", "after 11 AM")
        assert result["memory_type"] == "semantic"
        assert result["sub_type"] == "preference"

    def test_episodic_event(self):
        result = classify_memory("recent_trip", "went to Delhi last week")
        assert result["memory_type"] == "episodic"
        assert result["importance_score"] <= 6.0

    def test_procedural_instruction(self):
        result = classify_memory("response_style", "always use bullet points")
        assert result["memory_type"] == "procedural"
        assert result["importance_score"] >= 7.0

    def test_temporary_low_importance(self):
        result = classify_memory("temp_note", "check email today")
        assert result["importance_score"] <= 4.0

    def test_batch_classification(self):
        memories = [
            {"key": "name",      "value": "Ayush"},
            {"key": "diet",      "value": "vegetarian"},
            {"key": "trip",      "value": "went to Mumbai"},
        ]
        results = classify_memories(memories)
        assert len(results) == 3
        for r in results:
            assert "memory_type" in r
            assert "sub_type" in r
            assert "importance_score" in r

    def test_confidence_range(self):
        result = classify_memory("name", "Ayush")
        assert 1.0 <= result["importance_score"] <= 10.0