from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.retrieval.retriever import get_relevant_memories
from src.extractor.extractor import extract_memories
# from src.storage.vector_store import save_memories, clear_all_memories
from src.storage.vector_store import save_memories
from dotenv import load_dotenv
import os

load_dotenv()

# LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1000,
)

BASE_SYSTEM_PROMPT = """You are a helpful AI assistant with long-term memory.
Use the remembered information naturally in your responses.
Do not explicitly say 'I remember' or 'you told me' — just use the info naturally."""


def process_turn(user_message: str, turn_number: int, conversation_history: list) -> dict:
    """
    Full pipeline for a single conversation turn.
    1. Extract new memories from user message
    2. Save new memories to vector store
    3. Retrieve relevant memories for current query
    4. Inject memories into system prompt
    5. Get LLM response
    """

    # Step 1 — Extract new memories
    new_memories = extract_memories(user_message, turn_number)

    # Step 2 — Save new memories
    if new_memories:
        save_memories(new_memories)
        print(f"[Injector] Saved {len(new_memories)} new memories from turn {turn_number}")

    # Step 3 — Retrieve relevant memories AFTER saving
    memory_block = get_relevant_memories(user_message)

    # Step 4 — Build memory-augmented system prompt
    # if memory_block:
    #     system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{memory_block}"
    # else:
    #     system_prompt = BASE_SYSTEM_PROMPT
    memory_data = get_relevant_memories(user_message)
    memory_block = "\n".join([f"- {m['key']}: {m['value']}" for m in memory_data["memories"]])
    if memory_block:
       system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n[Remembered about this user:]\n{memory_block}"
    else:
       system_prompt = BASE_SYSTEM_PROMPT

    # Step 5 — Build messages and get LLM response
    messages = [SystemMessage(content=system_prompt)]
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": response.content})

    return {
        "turn": turn_number,
        "response": response.content,
        "active_memories": memory_block,
        "new_memories_saved": len(new_memories)
    }


# Quick test
if __name__ == "__main__":
    # Clear chroma_db folder before running
    # clear_all_memories()

    conversation_history = []

    turns = [
    "Hi! My name is Ayush and I prefer calls after 11 AM.",
    "I am vegetarian by the way.",
    "What is the capital of France?",
    "How do I write a for loop in Python?",
    "My manager is Priya and I need to submit the report by Friday.",  # ← add this
    "Can you call me tomorrow?",
    "Suggest me a good lunch.",
    "What is my name?",   # ← add comma
    "Where do I live?",
    ]

    print("=" * 60)
    print("SIMULATING LONG CONVERSATION WITH MEMORY")
    print("=" * 60)

    for i, message in enumerate(turns, start=1):
        print(f"\n[Turn {i}] User: {message}")
        result = process_turn(message, i, conversation_history)
        print(f"[Turn {i}] Assistant: {result['response']}")
        print(f"[Turn {i}] New memories saved: {result['new_memories_saved']}")