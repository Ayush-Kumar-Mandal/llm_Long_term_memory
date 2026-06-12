# System Architecture

## Overview

```
User Message (Turn N)
        │
        ▼
┌─────────────────────┐
│   Memory Extractor  │  ← Identifies facts, preferences, entities
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Memory Store     │  ← Vector DB + Structured DB
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Retrieval Engine   │  ← Semantic search on current query
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Context Builder   │  ← Injects top-K memories into prompt
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    LLM Inference    │
└────────┬────────────┘
         │
         ▼
      Response + active_memories
```

## Components

### 1. Memory Extractor (`src/extractor/`)
- Parses each turn for memorizable content
- Uses LLM or classifier to assign type + confidence
- Deduplicates and merges with existing memories

### 2. Memory Store (`src/storage/`)
- **Vector DB**: stores embeddings for semantic search
- **Structured DB**: stores metadata, type, key, value, turn info

### 3. Retrieval Engine (`src/retrieval/`)
- Embeds the current query
- Performs cosine similarity search
- Ranks by relevance + recency + confidence

### 4. Context Injector (`src/injector/`)
- Assembles compact memory block
- Injects into system prompt
- Tracks which memories were used

### 5. API (`src/api/`)
- FastAPI REST interface
- `/chat` — main conversation endpoint
- `/memories` — view/manage stored memories
