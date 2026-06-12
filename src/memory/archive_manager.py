"""
ArchiveManager — moves low-value memories to cold storage.
Archived memories are excluded from normal retrieval
but remain searchable as fallback.
"""

import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session

from src.storage.db import (
    MemoryRecord, ArchivedMemory, MemoryStatus,
    SessionLocal, init_db
)

logger = logging.getLogger(__name__)


class ArchiveManager:
    """
    Manages archival and retrieval of low-value memories.
    """

    def __init__(self, archive_threshold: float = 2.0):
        self.archive_threshold = archive_threshold
        init_db()
        logger.info("[ArchiveManager] Initialized. Threshold: %s", archive_threshold)

    def archive_memory(self, db: Session, memory: MemoryRecord) -> ArchivedMemory:
        """
        Move a single memory to cold storage (archived_memories table).
        """
        archived = ArchivedMemory(
            memory_id      = memory.memory_id,
            content        = memory.content,
            memory_type    = memory.memory_type,
            key            = memory.key,
            value          = memory.value,
            original_score = memory.decayed_score or 0.0,
            archived_at    = datetime.utcnow(),
            source_turn    = memory.source_turn
        )

        db.add(archived)
        memory.status = MemoryStatus.ARCHIVED
        db.commit()

        logger.info("[ArchiveManager] Archived memory: %s (%s=%s)", memory.memory_id, memory.key, memory.value)
        return archived

    def archive_batch(self, db: Session) -> int:
        """
        Archive all memories below the threshold score.
        Returns number of memories archived.
        """
        to_archive = db.query(MemoryRecord).filter(
            MemoryRecord.status       == MemoryStatus.ACTIVE,
            MemoryRecord.decayed_score < self.archive_threshold
        ).all()

        count = 0
        for memory in to_archive:
            self.archive_memory(db, memory)
            count += 1

        logger.info("[ArchiveManager] Archived %d memories in batch.", count)
        return count

    def restore_memory(self, db: Session, memory_id: str) -> Optional[MemoryRecord]:
        """
        Restore an archived memory back to active status.
        """
        archived = db.query(ArchivedMemory).filter(
            ArchivedMemory.memory_id == memory_id
        ).first()

        if not archived:
            logger.warning("[ArchiveManager] Memory %s not found in archive.", memory_id)
            return None

        # Restore to active memories
        memory = db.query(MemoryRecord).filter(
            MemoryRecord.memory_id == memory_id
        ).first()

        if memory:
            memory.status         = MemoryStatus.ACTIVE
            memory.last_accessed_at = datetime.utcnow()
            db.delete(archived)
            db.commit()
            logger.info("[ArchiveManager] Restored memory: %s", memory_id)
            return memory

        return None

    def search_archive(self, db: Session, query_key: str = None, memory_type: str = None) -> List[ArchivedMemory]:
        """
        Search cold storage — used as fallback when primary retrieval fails.

        Args:
            query_key: Optional key to filter by
            memory_type: Optional type filter

        Returns:
            List of ArchivedMemory records
        """
        q = db.query(ArchivedMemory)

        if query_key:
            q = q.filter(ArchivedMemory.key.ilike(f"%{query_key}%"))
        if memory_type:
            q = q.filter(ArchivedMemory.memory_type == memory_type)

        results = q.order_by(ArchivedMemory.original_score.desc()).limit(10).all()
        logger.info("[ArchiveManager] Archive search returned %d results.", len(results))
        return results

    def get_archive_stats(self, db: Session) -> dict:
        """Return statistics about archived memories."""
        total     = db.query(ArchivedMemory).count()
        by_type   = {}
        for mtype in ["episodic", "semantic", "procedural", "temporary"]:
            by_type[mtype] = db.query(ArchivedMemory).filter(
                ArchivedMemory.memory_type == mtype
            ).count()

        return {
            "total_archived": total,
            "by_type":        by_type
        }


# Quick test
if __name__ == "__main__":
    from src.storage.db import init_db, SessionLocal, MemoryRecord
    from datetime import datetime, timedelta
    import json

    init_db()
    db = SessionLocal()

    # Seed low-score memories
    test_memories = [
        MemoryRecord(memory_id="mem_a01", content="temporary: old_note is check email today",  memory_type="temporary", key="old_note",   value="check email today",  importance_score=2.0, confidence=0.5, access_count=0, source_turn=1, decayed_score=0.8, created_at=datetime.utcnow()-timedelta(days=60)),
        MemoryRecord(memory_id="mem_a02", content="episodic: old_trip is went to Pune 2 months ago", memory_type="episodic", key="old_trip", value="went to Pune 2 months ago", importance_score=3.0, confidence=0.6, access_count=1, source_turn=5, decayed_score=1.5, created_at=datetime.utcnow()-timedelta(days=45)),
        MemoryRecord(memory_id="mem_a03", content="semantic: name is Ayush",                   memory_type="semantic",  key="name",       value="Ayush",              importance_score=9.0, confidence=1.0, access_count=20, source_turn=1, decayed_score=8.5, created_at=datetime.utcnow()-timedelta(days=5)),
    ]

    for m in test_memories:
        existing = db.query(MemoryRecord).filter(MemoryRecord.memory_id == m.memory_id).first()
        if not existing:
            db.add(m)
    db.commit()

    manager = ArchiveManager(archive_threshold=2.0)

    print("\n" + "="*60)
    print("ARCHIVE MANAGER TEST")
    print("="*60)

    # Archive low-score memories
    count = manager.archive_batch(db)
    print(f"\nArchived {count} memories.")

    # Stats
    stats = manager.get_archive_stats(db)
    print(f"\nArchive stats: {json.dumps(stats, indent=2)}")

    # Search archive
    results = manager.search_archive(db, memory_type="episodic")
    print(f"\nEpisodic memories in archive: {len(results)}")
    for r in results:
        print(f"  - {r.key} = {r.value} (score: {r.original_score})")

    db.close()