# API Reference

Base URL: `http://localhost:8000`

## POST /chat
Send a conversation turn and get a memory-augmented response.

**Request:**
```json
{
  "session_id": "session_abc123",
  "turn": 42,
  "message": "Can you call me tomorrow?"
}
```

**Response:**
```json
{
  "response": "Sure! I'll call you after 11 AM as you prefer.",
  "active_memories": [...],
  "turn": 42
}
```

## GET /memories/{session_id}
Retrieve all stored memories for a session.

## DELETE /memories/{session_id}/{memory_id}
Delete a specific memory.

## GET /health
Health check endpoint.
