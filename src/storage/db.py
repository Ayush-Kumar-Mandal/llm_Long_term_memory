"""
SQLAlchemy models for structured memory metadata storage.
Supports O(log n) operations via indexed columns.
"""

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Boolean, DateTime, Text, Index, Enum as SAEnum
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os
import enum
import logging

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./memory.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Enums ─────────────────────────────────────────────────────

class MemoryStatus(str, enum.Enum):
    ACTIVE   = "active"
    ARCHIVED = "archived"
    DELETED  = "deleted"
    COMPRESSED = "compressed"

class MemoryAction(str, enum.Enum):
    KEEP      = "KEEP"
    REINFORCE = "REINFORCE"
    COMPRESS  = "COMPRESS"
    ARCHIVE   = "ARCHIVE"
    DELETE    = "DELETE"


# ── SQLAlchemy Models ─────────────────────────────────────────

class MemoryRecord(Base):
    """Primary memory metadata table."""
    __tablename__ = "memories"

    memory_id        = Column(String(32),  primary_key=True, index=True)
    content          = Column(Text,        nullable=False)
    memory_type      = Column(String(32),  nullable=False, index=True)
    sub_type         = Column(String(32),  nullable=True)
    key              = Column(String(128), nullable=False, index=True)
    value            = Column(Text,        nullable=False)

    # Scoring fields
    importance_score  = Column(Float,   default=5.0)
    retention_score   = Column(Float,   default=5.0)
    decayed_score     = Column(Float,   default=5.0)
    confidence        = Column(Float,   default=0.8)

    # Access tracking
    access_count      = Column(Integer, default=0,   index=True)
    source_turn       = Column(Integer, default=0)
    last_used_turn    = Column(Integer, default=0)

    # Timestamps
    created_at        = Column(DateTime, default=datetime.utcnow, index=True)
    last_accessed_at  = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Status
    status            = Column(SAEnum(MemoryStatus), default=MemoryStatus.ACTIVE, index=True)
    is_compressed     = Column(Boolean, default=False)
    compressed_from   = Column(Text,    nullable=True)  # JSON list of source memory_ids

    # Indexes for O(log n) operations
    __table_args__ = (
        Index("idx_status_decayed", "status", "decayed_score"),
        Index("idx_type_status",    "memory_type", "status"),
        Index("idx_last_accessed",  "last_accessed_at", "status"),
    )

    def __repr__(self):
        return f"<Memory {self.memory_id}: {self.key}={self.value} [{self.memory_type}]>"


class ArchivedMemory(Base):
    """Cold storage for archived memories."""
    __tablename__ = "archived_memories"

    memory_id        = Column(String(32),  primary_key=True)
    content          = Column(Text,        nullable=False)
    memory_type      = Column(String(32),  nullable=False)
    key              = Column(String(128), nullable=False)
    value            = Column(Text,        nullable=False)
    original_score   = Column(Float,       default=0.0)
    archived_at      = Column(DateTime,    default=datetime.utcnow)
    source_turn      = Column(Integer,     default=0)


class DecayLog(Base):
    """Audit log for decay operations."""
    __tablename__ = "decay_logs"

    id           = Column(Integer,    primary_key=True, autoincrement=True)
    memory_id    = Column(String(32), nullable=False, index=True)
    action       = Column(SAEnum(MemoryAction), nullable=False)
    score_before = Column(Float,      nullable=False)
    score_after  = Column(Float,      nullable=False)
    reason       = Column(String(256),nullable=True)
    logged_at    = Column(DateTime,   default=datetime.utcnow)


# ── DB Init ───────────────────────────────────────────────────

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] Tables created.")


def get_db():
    """Dependency for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("[DB] Initialized successfully.")
    print(f"[DB] Using: {DATABASE_URL}")