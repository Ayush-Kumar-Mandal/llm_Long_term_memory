"""
API Routes — clean separation of route definitions from main.py
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from src.injector.injector import process_turn
from src.storage.vector_store import get_all_memories, retrieve_memories
from src.storage.db import SessionLocal, MemoryRecord, MemoryStatus
from src.memory.decay_engine import MemoryDecayEngine
from src.memory.archive_manager import ArchiveManager
from src.storage.schemas import (
    ChatRequest, ChatResponse, DecayConfig,
    DecayReport, RetrievalResponse, RetrievalResult
)
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store
sessions = {}


# ── Chat ─────────────────────────────────────────────────────

@router.post("/chat", response_model=dict)
def chat(req: ChatRequest):
    """
    Send a message and get a memory-augmented response.
    Automatically extracts, stores, and retrieves memories.
    """
    if req.session_id not in sessions:
        sessions[req.session_id] = {"history": [], "turn": 0}

    session = sessions[req.session_id]
    session["turn"] += 1
    turn_number = session["turn"]

    try:
        result = process_turn(
            user_message=req.message,
            turn_number=turn_number,
            conversation_history=session["history"]
        )
    except Exception as e:
        logger.error("[Routes] Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "session_id":         req.session_id,
        "turn":               turn_number,
        "response":           result["response"],
        "new_memories_saved": result["new_memories_saved"],
        "active_memories":    result["active_memories"],
    }


# ── Memories ──────────────────────────────────────────────────

@router.get("/memories")
def get_memories(memory_type: str = None):
    """Get all stored memories, optionally filtered by type."""
    memories = get_all_memories(memory_type=memory_type)
    return {
        "total":       len(memories),
        "memory_type": memory_type or "all",
        "memories":    memories
    }


@router.get("/memories/search")
def search_memories(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    memory_type: str = Query(None, description="Filter by type: episodic/semantic/procedural")
):
    """Semantic + keyword hybrid search across memory collections."""
    from src.retrieval.retriever import get_relevant_memories
    results = get_relevant_memories(query, top_k=top_k, memory_type=memory_type)
    return results


@router.get("/memories/stats")
def memory_stats():
    """Get memory statistics across all collections."""
    db = SessionLocal()
    try:
        total   = db.query(MemoryRecord).count()
        active  = db.query(MemoryRecord).filter(MemoryRecord.status == MemoryStatus.ACTIVE).count()
        archived = db.query(MemoryRecord).filter(MemoryRecord.status == MemoryStatus.ARCHIVED).count()
        deleted  = db.query(MemoryRecord).filter(MemoryRecord.status == MemoryStatus.DELETED).count()

        by_type = {}
        for mtype in ["episodic", "semantic", "procedural"]:
            by_type[mtype] = db.query(MemoryRecord).filter(
                MemoryRecord.memory_type == mtype
            ).count()

        return {
            "total":    total,
            "active":   active,
            "archived": archived,
            "deleted":  deleted,
            "by_type":  by_type
        }
    finally:
        db.close()


# ── Sessions ──────────────────────────────────────────────────

@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get session info including turn count and history length."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return {
        "session_id":    session_id,
        "turn":          session["turn"],
        "history_length": len(session["history"])
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session and its conversation history."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted."}


# ── Decay ─────────────────────────────────────────────────────

@router.post("/decay/run")
def run_decay(config: DecayConfig = None):
    """
    Manually trigger a decay cycle.
    Evaluates all memories and applies KEEP/REINFORCE/COMPRESS/ARCHIVE/DELETE.
    """
    db     = SessionLocal()
    engine = MemoryDecayEngine(config=config or DecayConfig())
    try:
        report = engine.run_decay_cycle(db)
        return report.model_dump(exclude={"results"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/decay/scores")
def get_decay_scores(limit: int = 20):
    """Get current decay scores for all memories."""
    db = SessionLocal()
    try:
        memories = db.query(MemoryRecord).filter(
            MemoryRecord.status == MemoryStatus.ACTIVE
        ).order_by(MemoryRecord.decayed_score.desc()).limit(limit).all()

        return {
            "memories": [
                {
                    "memory_id":     m.memory_id,
                    "key":           m.key,
                    "value":         m.value,
                    "memory_type":   m.memory_type,
                    "decayed_score": m.decayed_score,
                    "access_count":  m.access_count,
                    "status":        m.status
                }
                for m in memories
            ]
        }
    finally:
        db.close()


# ── Archive ───────────────────────────────────────────────────

@router.get("/archive")
def get_archive(memory_type: str = None):
    """Get all archived memories."""
    db      = SessionLocal()
    manager = ArchiveManager()
    try:
        results = manager.search_archive(db, memory_type=memory_type)
        return {
            "total": len(results),
            "memories": [
                {
                    "memory_id":     r.memory_id,
                    "key":           r.key,
                    "value":         r.value,
                    "memory_type":   r.memory_type,
                    "original_score": r.original_score,
                    "archived_at":   r.archived_at.isoformat()
                }
                for r in results
            ]
        }
    finally:
        db.close()


@router.post("/archive/restore/{memory_id}")
def restore_from_archive(memory_id: str):
    """Restore a memory from archive back to active."""
    db      = SessionLocal()
    manager = ArchiveManager()
    try:
        memory = manager.restore_memory(db, memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found in archive")
        return {"message": f"Memory {memory_id} restored.", "memory_id": memory_id}
    finally:
        db.close()