"""
benchmark.py — measures system performance across 1000 turns.
Tests latency, memory recall accuracy, and retrieval relevance.
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.injector.injector import process_turn
from src.storage.vector_store import clear_all_memories
from src.retrieval.retriever import get_relevant_memories


# ── Test Cases ────────────────────────────────────────────────

SEED_TURNS = [
    (1,  "My name is Ayush."),
    (2,  "I prefer calls after 11 AM."),
    (3,  "I am vegetarian."),
    (4,  "My manager is Priya."),
    (5,  "Always use bullet points in responses."),
]

FILLER_TURNS = [
    "What is the capital of France?",
    "How do I write a for loop in Python?",
    "What is machine learning?",
    "Explain REST APIs.",
    "What is the Pythagorean theorem?",
]

RECALL_TESTS = [
    ("Can you call me tomorrow?",       "call_time",      "after 11 AM"),
    ("What should I eat for lunch?",    "diet",           "vegetarian"),
    ("Who is my manager?",              "manager",        "Priya"),
    ("What is my name?",                "name",           "Ayush"),
    ("How should you respond to me?",   "response_style", "bullet points"),
]


def run_benchmark(num_turns: int = 100):
    print("=" * 60)
    print(f"BENCHMARK — {num_turns} turns")
    print("=" * 60)

    clear_all_memories()
    conversation_history = []
    latencies = []

    # Phase 1 — Seed memories
    print("\n[Phase 1] Seeding memories...")
    for turn_num, message in SEED_TURNS:
        start = time.time()
        process_turn(message, turn_num, conversation_history)
        latencies.append(time.time() - start)

    # Phase 2 — Filler turns
    print(f"[Phase 2] Running {num_turns} filler turns...")
    for i in range(num_turns):
        turn_num = len(SEED_TURNS) + i + 1
        message  = FILLER_TURNS[i % len(FILLER_TURNS)]
        start    = time.time()
        process_turn(message, turn_num, conversation_history)
        latencies.append(time.time() - start)

        if (i + 1) % 50 == 0:
            avg = sum(latencies[-50:]) / 50
            print(f"  Turn {turn_num}: avg latency last 50 turns = {avg:.3f}s")

    # Phase 3 — Recall tests
    print("\n[Phase 3] Memory recall tests...")
    recall_results = []

    for query, expected_key, expected_value in RECALL_TESTS:
        start   = time.time()
        result  = get_relevant_memories(query, top_k=5)
        elapsed = time.time() - start

        # Check if expected value appears in results
        found = any(
            expected_value.lower() in m.get("value", "").lower()
            for m in result.get("memories", [])
        )

        recall_results.append({
            "query":          query,
            "expected":       f"{expected_key}={expected_value}",
            "found":          found,
            "latency_ms":     round(elapsed * 1000, 2),
            "memories_found": result.get("count", 0)
        })

        status = "✅ PASS" if found else "❌ FAIL"
        print(f"  {status} | '{query[:40]}' → {elapsed*1000:.1f}ms")

    # Phase 4 — Report
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)

    total_latency = sum(latencies)
    avg_latency   = total_latency / len(latencies)
    p95_latency   = sorted(latencies)[int(len(latencies) * 0.95)]
    recall_rate   = sum(1 for r in recall_results if r["found"]) / len(recall_results)

    print(f"Total turns        : {len(latencies)}")
    print(f"Avg latency        : {avg_latency*1000:.1f}ms")
    print(f"P95 latency        : {p95_latency*1000:.1f}ms")
    print(f"Memory recall rate : {recall_rate*100:.1f}%")
    print(f"Recall tests       : {sum(1 for r in recall_results if r['found'])}/{len(recall_results)} passed")

    return {
        "total_turns":    len(latencies),
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "p95_latency_ms": round(p95_latency * 1000, 2),
        "recall_rate":    round(recall_rate * 100, 2),
        "recall_results": recall_results
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=100, help="Number of filler turns")
    args = parser.parse_args()

    report = run_benchmark(num_turns=args.turns)
    print("\nFull report:")
    print(json.dumps(report, indent=2))