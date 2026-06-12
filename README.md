# 🧠 Long-Form Memory System

> Build a real-time long-form memory system that ensures information introduced in the first turn can be accurately recalled and applied in the 100th or 1,000th turn, without replaying the full conversation and without increasing system latency.

---



## 🏗️ System Architecture

```
User Message (Turn N)
        │
        ▼
┌─────────────────────────┐
│    Memory Extractor     │  ← LLM extracts facts, preferences, events
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Memory Classifier    │  ← Classifies into Episodic/Semantic/Procedural
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│              Memory Store                   │
│  ┌─────────────┐ ┌──────────┐ ┌──────────┐ │
│  │  Episodic   │ │ Semantic │ │Procedural│ │  ← 3 ChromaDB collections
│  │  (events)   │ │ (facts)  │ │ (rules)  │ │
│  └─────────────┘ └──────────┘ └──────────┘ │
│              + SQLite metadata              │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│    Hybrid Retriever     │  ← BM25 keyword + Semantic + RRF fusion
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Context Builder      │  ← Injects memories into system prompt
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      LLM Inference      │  ← Groq (llama-3.1-8b-instant)
└────────────┬────────────┘
             │
             ▼
     Response + active_memories
```

---

## 🧩 Three-Layer Memory Model

| Layer | Type | Examples | Decay Rate |
|-------|------|---------|------------|
| 🔵 **Semantic** | Facts, preferences, entities | "Name is Ayush", "Prefers calls after 11 AM" | Slow |
| 🟡 **Episodic** | Past events, experiences | "Went to Delhi last week" | Moderate |
| 🟢 **Procedural** | Behavior rules, instructions | "Always use bullet points" | Very slow |

---

## 🔍 Hybrid Retrieval (BM25 + Semantic + RRF)

```
Query: "Can you call me tomorrow?"
         │
         ├── Semantic Search  →  "call_time: after 11 AM"  ✅
         │   (ChromaDB)
         │
         ├── Keyword Search   →  "call" keyword match      ✅
         │   (BM25)
         │
         └── RRF Fusion       →  Best of both worlds       🎯
```

---

## ⚙️ Memory Decay Engine

Implements production-grade memory lifecycle management:

```
retention = w1*importance + w2*recency + w3*frequency + w4*relevance
decayed   = retention * exp(-λ * age_in_days)
```

### Memory Actions
| Action | Condition |
|--------|-----------|
| **KEEP** | Healthy score |
| **REINFORCE** | Frequently accessed |
| **COMPRESS** | Similar memories merged |
| **ARCHIVE** | Low score → cold storage |
| **DELETE** | Very low score |

### Decay Rates by Type
| Type | Decay Rate |
|------|-----------|
| Identity/Name | Extremely slow |
| Procedural rules | Very slow |
| Preferences | Slow |
| Episodic events | Moderate |
| Temporary notes | Aggressive |

---

## 📁 Project Structure

```
long-form-memory/
├── src/
│   ├── extractor/
│   │   ├── extractor.py        # Extracts memories from turns
│   │   └── classifier.py       # Classifies into 3 memory types
│   ├── storage/
│   │   ├── vector_store.py     # ChromaDB — 3 collections
│   │   ├── db.py               # SQLAlchemy models
│   │   ├── models.py           # Pydantic memory models
│   │   └── schemas.py          # API schemas
│   ├── retrieval/
│   │   └── retriever.py        # Hybrid BM25 + Semantic + RRF
│   ├── injector/
│   │   ├── injector.py         # Full pipeline orchestration
│   │   └── context_builder.py  # Memory-augmented prompt builder
│   ├── memory/
│   │   ├── decay_engine.py     # Memory decay & scoring
│   │   ├── compressor.py       # Memory compression
│   │   └── archive_manager.py  # Cold storage management
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── routes.py           # API endpoints
│       ├── schemas.py          # Request/response models
│       └── llm.py              # LLM connection (Groq)
├── tests/
│   ├── test_storage.py         # Storage tests 
│   ├── test_retrieval.py       # Retrieval tests 
│   ├── test_injector.py        # Injector tests 
│   └── test_e2e.py             # E2E 1000-turn tests 
├── scripts/
│   ├── demo_conversation.py    # Live demo
│   ├── benchmark.py            # Latency benchmarking
│   └── seed_memory.py          # Pre-load test memories
├── configs/
│   ├── config.yaml             # System configuration
│   └── prompts.yaml            # LLM prompts
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/long-form-memory.git
cd long-form-memory
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY
```

### 3. Seed Memories

```bash
python scripts/seed_memory.py
```

### 4. Run the API

```bash
python -m uvicorn src.api.main:app --reload
```

### 5. Open API Docs

```
http://localhost:8000/docs
```

### 6. Run Demo

```bash
python scripts/demo_conversation.py
```

### 7. Run Tests

```bash
pytest tests/ -v
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send message, get memory-augmented response |
| `GET` | `/memories` | View all stored memories |
| `GET` | `/memories/search` | Hybrid search memories |
| `GET` | `/memories/stats` | Memory statistics |
| `POST` | `/decay/run` | Trigger decay cycle |
| `GET` | `/decay/scores` | View decay scores |
| `GET` | `/archive` | View archived memories |
| `POST` | `/archive/restore/{id}` | Restore archived memory |

---

## 🧩 Memory Schema

```json
{
  "memory_id": "mem_0142",
  "memory_type": "semantic",
  "sub_type": "preference",
  "key": "call_time",
  "value": "after 11 AM",
  "confidence": 0.94,
  "importance_score": 7.5,
  "source_turn": 1,
  "last_used_turn": 937,
  "decayed_score": 7.2,
  "access_count": 12
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Vector DB | ChromaDB (3 collections) |
| Structured DB | SQLite + SQLAlchemy |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Keyword Search | BM25 (rank-bm25) |
| API | FastAPI + Uvicorn |
| Framework | LangChain |
| Testing | Pytest (31/31 passing) |

---
## end
