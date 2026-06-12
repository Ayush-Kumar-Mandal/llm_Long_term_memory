from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid


# ── Memory Types ─────────────────────────────────────────────

MemoryType = Literal["episodic", "semantic", "procedural"]

EpisodicSubType  = Literal["event", "experience", "conversation"]
SemanticSubType  = Literal["fact", "preference", "entity", "constraint", "commitment"]
ProceduralSubType = Literal["instruction", "style", "behavior"]


# ── Base Memory Model ─────────────────────────────────────────

class BaseMemory(BaseModel):
    memory_id:   str      = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    memory_type: MemoryType
    key:         str
    value:       str
    confidence:  float    = Field(ge=0.0, le=1.0)
    source_turn: int
    last_used_turn: int   = Field(default=0)
    created_at:  str      = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_content(self) -> str:
        """Human readable string for embedding."""
        return f"{self.memory_type}: {self.key} is {self.value}"

    def to_dict(self) -> dict:
        return self.model_dump()


# ── Episodic Memory ───────────────────────────────────────────

class EpisodicMemory(BaseMemory):
    """
    Episodic Memory — specific events and experiences.
    Examples:
      - "User went to Delhi last week"
      - "User mentioned they had a meeting with Priya"
      - "User was frustrated about the project deadline"
    """
    memory_type: MemoryType    = "episodic"
    sub_type:    EpisodicSubType = "event"
    context:     Optional[str] = None   # extra context about the event


# ── Semantic Memory ───────────────────────────────────────────

class SemanticMemory(BaseMemory):
    """
    Semantic Memory — facts, preferences, entities, constraints.
    Examples:
      - "User's name is Ayush"
      - "User prefers calls after 11 AM"
      - "User is vegetarian"
      - "User's manager is Priya"
    """
    memory_type: MemoryType    = "semantic"
    sub_type:    SemanticSubType = "fact"


# ── Procedural Memory ─────────────────────────────────────────

class ProceduralMemory(BaseMemory):
    """
    Procedural Memory — instructions and behavior rules.
    Examples:
      - "Always respond in bullet points"
      - "User prefers formal tone"
      - "Never suggest non-vegetarian food"
    """
    memory_type:  MemoryType      = "procedural"
    sub_type:     ProceduralSubType = "instruction"
    is_permanent: bool            = True   # procedural memories rarely expire


# ── Memory Factory ────────────────────────────────────────────

def create_memory(raw: dict) -> BaseMemory:
    """
    Factory function — creates the correct memory type
    from a raw dict returned by the extractor.
    """
    memory_type = raw.get("memory_type", "semantic")

    base = {
        "key":         raw["key"],
        "value":       raw["value"],
        "confidence":  raw.get("confidence", 0.8),
        "source_turn": raw.get("source_turn", 0),
        "last_used_turn": raw.get("source_turn", 0),
    }

    if memory_type == "episodic":
        return EpisodicMemory(
            **base,
            sub_type=raw.get("sub_type", "event"),
            context=raw.get("context")
        )
    elif memory_type == "procedural":
        return ProceduralMemory(
            **base,
            sub_type=raw.get("sub_type", "instruction")
        )
    else:
        return SemanticMemory(
            **base,
            sub_type=raw.get("sub_type", "fact")
        )


# ── Quick Test ────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Test all three memory types
    test_memories = [
        {
            "memory_type": "episodic",
            "key": "recent_event",
            "value": "went to Delhi last week",
            "confidence": 0.9,
            "source_turn": 5,
            "sub_type": "experience"
        },
        {
            "memory_type": "semantic",
            "key": "call_time",
            "value": "after 11 AM",
            "confidence": 0.95,
            "source_turn": 1,
            "sub_type": "preference"
        },
        {
            "memory_type": "procedural",
            "key": "response_style",
            "value": "always use bullet points",
            "confidence": 1.0,
            "source_turn": 2,
            "sub_type": "style"
        }
    ]

    print("=" * 60)
    print("MEMORY MODELS TEST")
    print("=" * 60)

    for raw in test_memories:
        memory = create_memory(raw)
        print(f"\n[{memory.memory_type.upper()}]")
        print(json.dumps(memory.to_dict(), indent=2))
        print(f"Embedded as: '{memory.to_content()}'")