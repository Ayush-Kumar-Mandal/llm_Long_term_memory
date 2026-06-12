from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq LLM via Langchain
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1000,
)

def chat_with_llm(messages: list, system_prompt: str = "") -> str:
    """
    Send messages to Groq LLM via Langchain and get a response.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."}
        system_prompt: Optional system instructions

    Returns:
        str: LLM response text
    """
    langchain_messages = []

    if system_prompt:
        langchain_messages.append(SystemMessage(content=system_prompt))

    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    response = llm.invoke(langchain_messages)
    return response.content


# Quick test
if __name__ == "__main__":
    reply = chat_with_llm(
        messages=[{"role": "user", "content": "Say hello!"}],
        system_prompt="You are a helpful assistant."
    )
    print("LLM Response:", reply)