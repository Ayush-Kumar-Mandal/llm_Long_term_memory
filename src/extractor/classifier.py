"""
Memory Classifier — classifies extracted memories into:
- episodic / semantic / procedural
- sub_type
- importance_score (1-10) for decay engine
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
import logging

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=500,
)

CLASSIFIER_PROMPT = """
You are a memory classifier for an AI assistant.
Classify the given memory into one of these 3 types:

1. EPISODIC — specific events or experiences that happened
   Examples: "went to Delhi last week", "had a meeting with Priya"
   Sub-types: event, experience, conversation

2. SEMANTIC — facts, preferences, entities, constraints about the user
   Examples: "name is Ayush", "prefers calls after 11 AM", "is vegetarian"
   Sub-types: fact, preference, entity, constraint, commitment

3. PROCEDURAL — instructions and behavior rules for the assistant
   Examples: "always use bullet points", "respond formally"
   Sub-types: instruction, style, behavior

Also assign an importance_score from 1-10:
- Identity facts (name, age):        9-10
- Procedural instructions:           8-9
- Long-term preferences:             7-8
- Goals and commitments:             7-8
- Entities (manager, friends):       6-7
- Constraints (diet, allergies):     6-7
- Recent events:                     4-5
- Temporary notes:                   1-3

Memory to classify:
key: KEY_HERE
value: VALUE_HERE

Return ONLY a JSON object:
{
  "memory_type": "semantic",
  "sub_type": "preference",
  "importance_score": 7.5
}
"""


def classify_memory(key: str, value: str) -> dict:
    """
    Classify a single memory into type, sub_type and importance_score.

    Args:
        key: Memory key
        value: Memory value

    Returns:
        dict with memory_type, sub_type, importance_score
    """
    prompt = CLASSIFIER_PROMPT.replace("KEY_HERE", key).replace("VALUE_HERE", value)

    response = llm.invoke([
        SystemMessage(content="You are a memory classifier. Return valid JSON only."),
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
        return {
            "memory_type":     result.get("memory_type",     "semantic"),
            "sub_type":        result.get("sub_type",        "fact"),
            "importance_score": float(result.get("importance_score", 5.0))
        }
    except json.JSONDecodeError:
        logger.warning("[Classifier] Failed to parse: %s", raw)
        return {
            "memory_type":     "semantic",
            "sub_type":        "fact",
            "importance_score": 5.0
        }


def classify_memories(memories: list) -> list:
    """
    Classify a list of memories and add memory_type, sub_type, importance_score.

    Args:
        memories: List of memory dicts from extractor

    Returns:
        List of memories with classification fields added
    """
    classified = []
    for mem in memories:
        result = classify_memory(mem["key"], mem["value"])
        mem["memory_type"]     = result["memory_type"]
        mem["sub_type"]        = result["sub_type"]
        mem["importance_score"] = result["importance_score"]
        classified.append(mem)
        logger.info(
            "[Classifier] %s=%s → %s/%s (importance: %s)",
            mem["key"], mem["value"],
            mem["memory_type"], mem["sub_type"],
            mem["importance_score"]
        )
    return classified


# Quick test
if __name__ == "__main__":
    test_memories = [
        {"key": "name",           "value": "Ayush"},
        {"key": "call_time",      "value": "after 11 AM"},
        {"key": "diet",           "value": "vegetarian"},
        {"key": "manager",        "value": "Priya"},
        {"key": "recent_trip",    "value": "went to Delhi last week"},
        {"key": "response_style", "value": "always use bullet points"},
        {"key": "report_due",     "value": "Friday"},
        {"key": "temp_note",      "value": "check email today"},
    ]

    print("=" * 60)
    print("CLASSIFIER TEST — with importance scores")
    print("=" * 60)

    results = classify_memories(test_memories)
    for m in results:
        print(f"\n  key:              {m['key']}")
        print(f"  value:            {m['value']}")
        print(f"  memory_type:      {m['memory_type']}")
        print(f"  sub_type:         {m['sub_type']}")
        print(f"  importance_score: {m['importance_score']}")