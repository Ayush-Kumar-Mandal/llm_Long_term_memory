from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.storage.models import BaseMemory, create_memory
from dotenv import load_dotenv
import uuid

load_dotenv()

# Shared embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

CHROMA_PATH = "./chroma_db"

# 3 separate collections — one per memory type
episodic_store = Chroma(
    collection_name="episodic_memory",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

semantic_store = Chroma(
    collection_name="semantic_memory",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

procedural_store = Chroma(
    collection_name="procedural_memory",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

# For backward compatibility
vector_store = semantic_store

STORE_MAP = {
    "episodic":   episodic_store,
    "semantic":   semantic_store,
    "procedural": procedural_store,
}


def get_store(memory_type: str) -> Chroma:
    """Return the correct store for a memory type."""
    return STORE_MAP.get(memory_type, semantic_store)


def save_memory(memory: dict) -> str:
    """
    Save a memory to the correct collection based on memory_type.
    Skips duplicates.
    """
    memory_type = memory.get("memory_type", "semantic")
    store = get_store(memory_type)

    content = f"{memory_type}: {memory['key']} is {memory['value']}"

    # Deduplication check
    existing = store.similarity_search(content, k=1)
    if existing:
        top = existing[0]
        if top.metadata.get("key") == memory["key"] and \
           top.metadata.get("value") == memory["value"]:
            print(f"[Storage] Skipping duplicate: {memory['key']} = {memory['value']}")
            return top.metadata.get("memory_id", "existing")

    memory_id = f"mem_{uuid.uuid4().hex[:8]}"

    doc = Document(
        page_content=content,
        metadata={
            "memory_id":   memory_id,
            "memory_type": memory_type,
            "sub_type":    memory.get("sub_type", "fact"),
            "key":         memory["key"],
            "value":       memory["value"],
            "confidence":  memory.get("confidence", 0.8),
            "source_turn": memory.get("source_turn", 0),
            "last_used_turn": memory.get("source_turn", 0),
        }
    )

    store.add_documents([doc], ids=[memory_id])
    print(f"[Storage] [{memory_type.upper()}] Saved: {memory_id} → {content}")
    return memory_id


def save_memories(memories: list) -> list:
    """Save a list of memories."""
    return [save_memory(m) for m in memories]


def retrieve_memories(query: str, top_k: int = 5, memory_type: str = None) -> list:
    """
    Retrieve memories from one or all collections.

    Args:
        query: Search query
        top_k: Number of results
        memory_type: If set, search only that collection. If None, search all.
    """
    if memory_type:
        store = get_store(memory_type)
        results = store.similarity_search(query, k=top_k)
        return [{"content": doc.page_content, **doc.metadata} for doc in results]

    # Search all 3 collections
    all_results = []
    for mtype, store in STORE_MAP.items():
        try:
            results = store.similarity_search(query, k=top_k)
            for doc in results:
                all_results.append({"content": doc.page_content, **doc.metadata})
        except Exception:
            pass

    return all_results


def get_all_memories(memory_type: str = None) -> list:
    """Get all memories from one or all collections."""
    if memory_type:
        store = get_store(memory_type)
        results = store.get()
    else:
        # Merge all collections
        all_results = {"ids": [], "documents": [], "metadatas": []}
        for store in STORE_MAP.values():
            try:
                r = store.get()
                all_results["ids"].extend(r["ids"])
                all_results["documents"].extend(r["documents"])
                all_results["metadatas"].extend(r["metadatas"])
            except Exception:
                pass
        results = all_results

    memories = []
    for i, doc_id in enumerate(results["ids"]):
        memories.append({
            "memory_id": doc_id,
            "content":   results["documents"][i],
            "metadata":  results["metadatas"][i]
        })
    return memories


def clear_all_memories():
    """Clear all memories by resetting each collection."""
    try:
        episodic_store.reset_collection()
        semantic_store.reset_collection()
        procedural_store.reset_collection()
        print("[Storage] All memories cleared.")
    except Exception as e:
        print(f"[Storage] Clear failed: {e}")


# Quick test
if __name__ == "__main__":
    # clear_all_memories()

    test_memories = [
        {"memory_type": "semantic",   "sub_type": "preference", "key": "call_time",      "value": "after 11 AM",              "confidence": 0.95, "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "fact",       "key": "name",            "value": "Ayush",                    "confidence": 1.0,  "source_turn": 1},
        {"memory_type": "semantic",   "sub_type": "constraint", "key": "diet",            "value": "vegetarian",               "confidence": 1.0,  "source_turn": 2},
        {"memory_type": "episodic",   "sub_type": "event",      "key": "recent_trip",     "value": "went to Delhi last week",  "confidence": 0.9,  "source_turn": 3},
        {"memory_type": "procedural", "sub_type": "style",      "key": "response_style",  "value": "always use bullet points", "confidence": 1.0,  "source_turn": 4},
    ]

    print("\n--- Saving Memories ---")
    save_memories(test_memories)

    print("\n--- Retrieve from ALL collections ---")
    results = retrieve_memories("Can you call me tomorrow?", top_k=3)
    for r in results:
        print(f"  [{r.get('memory_type','?').upper()}] {r['key']} = {r['value']}")

    print("\n--- Retrieve EPISODIC only ---")
    results = retrieve_memories("trip", top_k=3, memory_type="episodic")
    for r in results:
        print(f"  [{r.get('memory_type','?').upper()}] {r['key']} = {r['value']}")

    print("\n--- Retrieve PROCEDURAL only ---")
    results = retrieve_memories("how to respond", top_k=3, memory_type="procedural")
    for r in results:
        print(f"  [{r.get('memory_type','?').upper()}] {r['key']} = {r['value']}")