"""
MemoryDecayEngine — production-grade memory decay and retention scoring.

Implements:
- Hybrid retention scoring (importance + recency + frequency + relevance)
- Exponential time decay
- Memory-type-aware decay rates
- Batch processing
- Memory actions: KEEP, REINFORCE, COMPRESS, ARCHIVE, DELETE
"""

import math
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session

from src.storage.db import MemoryRecord, DecayLog, MemoryStatus, MemoryAction, SessionLocal, init_db
from src.storage.schemas import DecayConfig, MemoryDecayResult, DecayReport

logger = logging.getLogger(__name__)


# ── Decay Rate per Memory Type ────────────────────────────────

DECAY_RATES = {
    # memory_type       : lambda multiplier (higher = faster decay)
    "identity":           0.001,   # extremely slow
    "procedural":         0.002,   # very slow
    "semantic":           0.005,   # slow
    "preference":         0.004,   # slow
    "goal":               0.003,   # slow
    "episodic":           0.02,    # moderate
    "event":              0.05,    # fast
    "temporary":          0.1,     # aggressive
}

# Importance score per memory type
DEFAULT_IMPORTANCE = {
    "identity":   9.0,
    "procedural": 8.0,
    "semantic":   6.0,
    "preference": 7.0,
    "goal":       8.0,
    "episodic":   5.0,
    "event":      4.0,
    "temporary":  2.0,
}


class MemoryDecayEngine:
    """
    Calculates retention scores and applies decay to memories.

    Retention Formula:
        retention = w1*importance + w2*recency + w3*frequency + w4*relevance
        decayed   = retention * exp(-lambda * age_in_days)
    """

    def __init__(self, config: DecayConfig = None):
        self.config = config or DecayConfig()
        init_db()
        logger.info("[DecayEngine] Initialized with config: %s", self.config)

    # ── Scoring Components ────────────────────────────────────

    def _importance_score(self, memory: MemoryRecord) -> float:
        """Normalized importance score (0-1)."""
        return memory.importance_score / 10.0

    def _recency_score(self, memory: MemoryRecord) -> float:
        """
        Recency score based on last access time.
        More recently accessed = higher score.
        """
        now = datetime.utcnow()
        last = memory.last_accessed_at or memory.created_at
        days_since = max((now - last).total_seconds() / 86400, 0)
        # Exponential decay: score = exp(-0.1 * days)
        return math.exp(-0.1 * days_since)

    def _frequency_score(self, memory: MemoryRecord) -> float:
        """
        Frequency score based on access count.
        More accesses = higher score, with diminishing returns.
        """
        count = memory.access_count or 0
        # Log scale with ceiling at 1.0
        return min(math.log1p(count) / math.log1p(100), 1.0)

    def _relevance_score(self, memory: MemoryRecord) -> float:
        """
        Relevance score — based on confidence and type importance.
        """
        type_importance = DEFAULT_IMPORTANCE.get(memory.memory_type, 5.0) / 10.0
        return (memory.confidence + type_importance) / 2.0

    # ── Retention Score ───────────────────────────────────────

    def compute_retention_score(self, memory: MemoryRecord) -> float:
        """
        Compute weighted retention score.
        retention = w1*importance + w2*recency + w3*frequency + w4*relevance
        """
        cfg = self.config
        score = (
            cfg.w1_importance * self._importance_score(memory) +
            cfg.w2_recency    * self._recency_score(memory)    +
            cfg.w3_frequency  * self._frequency_score(memory)  +
            cfg.w4_relevance  * self._relevance_score(memory)
        )
        # Normalize to 0-10
        return round(score * 10, 4)

    def compute_decayed_score(self, memory: MemoryRecord) -> float:
        """
        Apply exponential time decay to retention score.
        decayed = retention * exp(-lambda * age_in_days)
        """
        retention = self.compute_retention_score(memory)
        now = datetime.utcnow()
        created = memory.created_at or now
        age_in_days = max((now - created).total_seconds() / 86400, 0)

        # Use type-specific decay rate
        base_lambda = self.config.decay_lambda
        type_lambda = DECAY_RATES.get(memory.memory_type, base_lambda)
        effective_lambda = base_lambda * type_lambda * 100

        decayed = retention * math.exp(-effective_lambda * age_in_days)
        return round(decayed, 4)

    # ── Memory Actions ────────────────────────────────────────

    def determine_action(self, memory: MemoryRecord, decayed_score: float) -> tuple:
        """
        Determine what action to take based on decayed score.

        Returns:
            (MemoryAction, reason)
        """
        cfg = self.config

        # Reinforcement — recently accessed or high frequency
        if memory.access_count > 10 or (
            memory.last_accessed_at and
            (datetime.utcnow() - memory.last_accessed_at).days < 1
        ):
            return MemoryAction.REINFORCE, "High access frequency or recent access"

        # Delete — very low score
        if decayed_score < cfg.delete_threshold:
            return MemoryAction.DELETE, f"Score {decayed_score} below delete threshold {cfg.delete_threshold}"

        # Archive — low score
        if decayed_score < cfg.archive_threshold:
            return MemoryAction.ARCHIVE, f"Score {decayed_score} below archive threshold {cfg.archive_threshold}"

        # Compress — medium score, old memory
        now = datetime.utcnow()
        age_days = (now - (memory.created_at or now)).days
        if decayed_score < 5.0 and age_days > 7:
            return MemoryAction.COMPRESS, f"Old memory (age={age_days}d) with medium score"

        return MemoryAction.KEEP, f"Score {decayed_score} is healthy"

    # ── Reinforcement ─────────────────────────────────────────

    def reinforce(self, db: Session, memory: MemoryRecord) -> None:
        """Boost a memory's score on retrieval."""
        memory.access_count    += 1
        memory.last_accessed_at = datetime.utcnow()
        memory.importance_score = min(memory.importance_score + 0.5, 10.0)
        db.commit()
        logger.debug("[DecayEngine] Reinforced memory %s", memory.memory_id)

    # ── Batch Decay ───────────────────────────────────────────

    def run_decay_cycle(self, db: Session, batch_size: int = 500) -> DecayReport:
        """
        Run a full decay cycle on all active memories in batches.
        Avoids recomputing scores for all memories every query.
        """
        logger.info("[DecayEngine] Starting decay cycle...")

        results   = []
        kept      = reinforced = compressed = archived = deleted = 0
        offset    = 0

        while True:
            # Batch fetch active memories
            batch = (
                db.query(MemoryRecord)
                .filter(MemoryRecord.status == MemoryStatus.ACTIVE)
                .order_by(MemoryRecord.last_accessed_at)
                .offset(offset)
                .limit(batch_size)
                .all()
            )

            if not batch:
                break

            for memory in batch:
                score_before  = memory.decayed_score or 5.0
                decayed_score = self.compute_decayed_score(memory)
                action, reason = self.determine_action(memory, decayed_score)

                # Apply action
                if action == MemoryAction.REINFORCE:
                    self.reinforce(db, memory)
                    reinforced += 1
                elif action == MemoryAction.DELETE:
                    memory.status = MemoryStatus.DELETED
                    deleted += 1
                elif action == MemoryAction.ARCHIVE:
                    memory.status = MemoryStatus.ARCHIVED
                    archived += 1
                elif action == MemoryAction.COMPRESS:
                    memory.status = MemoryStatus.COMPRESSED
                    compressed += 1
                else:
                    kept += 1

                # Update scores
                memory.retention_score = self.compute_retention_score(memory)
                memory.decayed_score   = decayed_score
                memory.updated_at      = datetime.utcnow()

                # Log
                db.add(DecayLog(
                    memory_id    = memory.memory_id,
                    action       = action,
                    score_before = score_before,
                    score_after  = decayed_score,
                    reason       = reason
                ))

                results.append(MemoryDecayResult(
                    memory_id    = memory.memory_id,
                    action       = action,
                    score_before = score_before,
                    score_after  = decayed_score,
                    reason       = reason
                ))

            db.commit()
            offset += batch_size
            logger.info("[DecayEngine] Processed batch offset=%d", offset)

        report = DecayReport(
            total_evaluated = len(results),
            kept            = kept,
            reinforced      = reinforced,
            compressed      = compressed,
            archived        = archived,
            deleted         = deleted,
            results         = results
        )

        logger.info("[DecayEngine] Cycle complete: %s", report.model_dump(exclude={"results"}))
        return report

    def update_on_retrieval(self, db: Session, memory_id: str) -> None:
        """Call this whenever a memory is retrieved to reinforce it."""
        memory = db.query(MemoryRecord).filter(
            MemoryRecord.memory_id == memory_id
        ).first()
        if memory:
            self.reinforce(db, memory)


# Quick test
if __name__ == "__main__":
    from src.storage.db import init_db, SessionLocal, MemoryRecord
    from datetime import datetime, timedelta
    import json

    init_db()
    db = SessionLocal()

    # Seed test memories
    test_data = [
        MemoryRecord(memory_id="mem_001", content="name is Ayush",         memory_type="semantic",   key="name",          value="Ayush",              importance_score=8.0, confidence=1.0,  access_count=15, created_at=datetime.utcnow()-timedelta(days=1),  last_accessed_at=datetime.utcnow()-timedelta(hours=1)),
        MemoryRecord(memory_id="mem_002", content="call after 11 AM",      memory_type="preference", key="call_time",     value="after 11 AM",        importance_score=7.0, confidence=0.95, access_count=8,  created_at=datetime.utcnow()-timedelta(days=5),  last_accessed_at=datetime.utcnow()-timedelta(days=1)),
        MemoryRecord(memory_id="mem_003", content="went to Delhi",         memory_type="episodic",   key="recent_trip",   value="went to Delhi",      importance_score=4.0, confidence=0.8,  access_count=1,  created_at=datetime.utcnow()-timedelta(days=30), last_accessed_at=datetime.utcnow()-timedelta(days=25)),
        MemoryRecord(memory_id="mem_004", content="use bullet points",     memory_type="procedural", key="style",         value="bullet points",      importance_score=9.0, confidence=1.0,  access_count=20, created_at=datetime.utcnow()-timedelta(days=2),  last_accessed_at=datetime.utcnow()-timedelta(hours=2)),
        MemoryRecord(memory_id="mem_005", content="temporary note",        memory_type="temporary",  key="temp_note",     value="check email today",  importance_score=2.0, confidence=0.5,  access_count=0,  created_at=datetime.utcnow()-timedelta(days=60), last_accessed_at=datetime.utcnow()-timedelta(days=59)),
    ]

    # Clear and seed
    db.query(MemoryRecord).delete()
    db.query(DecayLog).delete()
    db.commit()
    for m in test_data:
        db.add(m)
    db.commit()

    # Run decay
    engine = MemoryDecayEngine()
    report = engine.run_decay_cycle(db)

    print("\n" + "="*60)
    print("DECAY CYCLE REPORT")
    print("="*60)
    print(f"Total evaluated : {report.total_evaluated}")
    print(f"Kept            : {report.kept}")
    print(f"Reinforced      : {report.reinforced}")
    print(f"Compressed      : {report.compressed}")
    print(f"Archived        : {report.archived}")
    print(f"Deleted         : {report.deleted}")
    print("\nDetailed Results:")
    for r in report.results:
        print(f"  {r.memory_id} → {r.action.value:10} | {r.score_before:.3f} → {r.score_after:.3f} | {r.reason}")

    db.close()