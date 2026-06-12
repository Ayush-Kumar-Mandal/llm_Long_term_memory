"""
Pydantic schemas for memory system — request/response validation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────

class MemoryTypeEnum(str, Enum):
    EPISODIC   = "episodic"
    SEMANTIC   = "semantic"
    PROCEDURAL = "procedural"

class MemoryStatusEnum(str, Enum):
    ACTIVE     = "active"
    ARCHIVED   = "archived"
    DELETED    = "deleted"
    COMPRESSED = "compressed"

class MemoryActionEnum(str, Enum):
    KEEP      = "KEEP"
    REINFORCE = "REINFORCE"
    COMPRESS  = "COMPRESS"
    ARCHIVE   = "ARCHIVE"
    DELETE    = "DELETE"


# ── Memory Schemas ────────────────────────────────────────────

class MemoryBase(BaseModel):
    memory_type:     MemoryTypeEnum
    sub_type:        Optional[str]  = None
    key:             str
    value:           str
    confidence:      float          = Field(default=0.8, ge=0.0, le=1.0)
    importance_score: float         = Field(default=5.0, ge=1.0, le=10.0)
    source_turn:     int            = 0


class MemoryCreate(MemoryBase):
    """Schema for creating a new memory."""
    pass


class MemoryResponse(MemoryBase):
    """Schema for reading a memory."""
    model_config = ConfigDict(from_attributes=True)

    memory_id:        str
    retention_score:  float
    decayed_score:    float
    access_count:     int
    last_used_turn:   int
    status:           MemoryStatusEnum
    created_at:       datetime
    last_accessed_at: datetime


class MemoryDecayResult(BaseModel):
    """Result of a decay evaluation for a single memory."""
    memory_id:       str
    action:          MemoryActionEnum
    score_before:    float
    score_after:     float
    reason:          str


class DecayReport(BaseModel):
    """Summary report of a decay cycle."""
    total_evaluated: int
    kept:            int
    reinforced:      int
    compressed:      int
    archived:        int
    deleted:         int
    results:         List[MemoryDecayResult]
    ran_at:          datetime = Field(default_factory=datetime.utcnow)


class CompressionResult(BaseModel):
    """Result of memory compression."""
    new_memory_id:    str
    source_memory_ids: List[str]
    compressed_content: str
    memories_reduced: int


class RetrievalResult(BaseModel):
    """Single memory retrieval result with ranking score."""
    memory_id:    str
    content:      str
    memory_type:  MemoryTypeEnum
    key:          str
    value:        str
    source_turn:  int
    ranking_score: float
    from_archive: bool = False


class RetrievalResponse(BaseModel):
    """Full retrieval response."""
    query:    str
    memories: List[RetrievalResult]
    count:    int
    searched_archive: bool = False


# ── API Schemas ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str


class ChatResponse(BaseModel):
    session_id:          str
    turn:                int
    response:            str
    new_memories_saved:  int
    active_memories:     List[RetrievalResult] = []


class DecayConfig(BaseModel):
    """Configurable weights for decay engine."""
    w1_importance:  float = Field(default=0.35, ge=0.0, le=1.0)
    w2_recency:     float = Field(default=0.25, ge=0.0, le=1.0)
    w3_frequency:   float = Field(default=0.20, ge=0.0, le=1.0)
    w4_relevance:   float = Field(default=0.20, ge=0.0, le=1.0)
    decay_lambda:   float = Field(default=0.01, ge=0.0)
    archive_threshold: float = Field(default=2.0)
    delete_threshold:  float = Field(default=0.5)


if __name__ == "__main__":
    import json

    # Test schemas
    mem = MemoryCreate(
        memory_type="semantic",
        key="call_time",
        value="after 11 AM",
        confidence=0.95,
        importance_score=7.0,
        source_turn=1
    )
    print("MemoryCreate:")
    print(json.dumps(mem.model_dump(), indent=2))

    cfg = DecayConfig()
    print("\nDecayConfig (defaults):")
    print(json.dumps(cfg.model_dump(), indent=2))