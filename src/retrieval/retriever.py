from src.storage.vector_store import episodic_store, semantic_store, procedural_store, STORE_MAP
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()


def get_all_docs(memory_type: str = None) -> list:
    """Fetch all documents from one or all collections."""
    docs = []

    stores = {memory_type: STORE_MAP[memory_type]} if memory_type else STORE_MAP

    for mtype, store in stores.items():
        try:
            results = store.get()
            for i, doc_id in enumerate(results["ids"]):
                docs.append({
                    "id":          doc_id,
                    "content":     results["documents"][i],
                    "metadata":    results["metadatas"][i],
                    "memory_type": mtype
                })
        except Exception:
            pass

    return docs


def semantic_search(query: str, top_k: int = 5, memory_type: str = None) -> list:
    """
    Semantic search across one or all ChromaDB collections.
    Returns list of (content, metadata, score) tuples.
    """
    results = []
    stores = {memory_type: STORE_MAP[memory_type]} if memory_type else STORE_MAP

    for mtype, store in stores.items():
        try:
            hits = store.similarity_search_with_score(query, k=top_k)
            for doc, score in hits:
                results.append((doc.page_content, doc.metadata, score))
        except Exception:
            pass

    # Sort by score ascending (lower = more similar in ChromaDB)
    results.sort(key=lambda x: x[2])
    return results[:top_k]


def keyword_search(query: str, docs: list, top_k: int = 5) -> list:
    """
    BM25 keyword search across all docs.
    Returns list of (content, metadata, score) tuples.
    """
    if not docs:
        return []

    tokenized_docs = [doc["content"].lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:top_k]
    return [(doc["content"], doc["metadata"], score) for score, doc in ranked]


def reciprocal_rank_fusion(semantic_results: list, keyword_results: list, k: int = 60) -> list:
    """
    Combine semantic + keyword results using Reciprocal Rank Fusion.
    Higher RRF score = more relevant.
    """
    semantic_ranks = {c: r + 1 for r, (c, _, _) in enumerate(semantic_results)}
    keyword_ranks  = {c: r + 1 for r, (c, _, _) in enumerate(keyword_results)}
    all_contents   = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

    meta_lookup = {}
    for content, metadata, _ in semantic_results + keyword_results:
        meta_lookup[content] = metadata

    rrf_scores = {}
    for content in all_contents:
        sem_rank = semantic_ranks.get(content, len(semantic_results) + k)
        kw_rank  = keyword_ranks.get(content, len(keyword_results) + k)
        rrf_scores[content] = (1 / (k + sem_rank)) + (1 / (k + kw_rank))

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(content, meta_lookup[content], score) for content, score in ranked]


def get_relevant_memories(query: str, top_k: int = 5, memory_type: str = None) -> dict:
    """
    Hybrid search across all 3 memory collections.
    Combines BM25 keyword + ChromaDB semantic + RRF fusion.

    Args:
        query: Current user message
        top_k: Number of memories to retrieve
        memory_type: Optional filter (episodic/semantic/procedural)

    Returns:
        dict with structured memory results
    """
    all_docs = get_all_docs(memory_type)

    if not all_docs:
        return {"query": query, "memories": [], "count": 0}

    sem_results = semantic_search(query, top_k=top_k, memory_type=memory_type)
    kw_results  = keyword_search(query, all_docs, top_k=top_k)
    fused       = reciprocal_rank_fusion(sem_results, kw_results)[:top_k]

    if not fused:
        return {"query": query, "memories": [], "count": 0}

    memories = []
    for content, metadata, score in fused:
        memories.append({
            "content":     content,
            "memory_type": metadata.get("memory_type", "semantic"),
            "sub_type":    metadata.get("sub_type", "fact"),
            "key":         metadata.get("key"),
            "value":       metadata.get("value"),
            "source_turn": metadata.get("source_turn"),
            "score":       round(score, 4)
        })

    return {
        "query":    query,
        "memories": memories,
        "count":    len(memories)
    }


def format_memories_for_prompt(memory_data: dict) -> str:
    """
    Format retrieved memories into a prompt block grouped by memory type.
    """
    memories = memory_data.get("memories", [])
    if not memories:
        return ""

    episodic   = [m for m in memories if m["memory_type"] == "episodic"]
    semantic   = [m for m in memories if m["memory_type"] == "semantic"]
    procedural = [m for m in memories if m["memory_type"] == "procedural"]

    lines = ["[Memory Context]"]

    if semantic:
        lines.append("Facts & Preferences:")
        for m in semantic:
            lines.append(f"  - {m['key']}: {m['value']}")

    if episodic:
        lines.append("Past Events:")
        for m in episodic:
            lines.append(f"  - {m['value']} (turn {m['source_turn']})")

    if procedural:
        lines.append("Behavior Rules:")
        for m in procedural:
            lines.append(f"  - {m['value']}")

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    import json

    test_queries = [
        "Can you call me tomorrow?",
        "What should I eat for lunch?",
        "Did I go anywhere recently?",
        "How should you respond to me?",
        "What is my name?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = get_relevant_memories(query, top_k=3)
        print(f"Found {result['count']} memories:")
        for m in result["memories"]:
            print(f"  [{m['memory_type'].upper()}] {m['key']} = {m['value']} (score: {m['score']})")
        print(format_memories_for_prompt(result))