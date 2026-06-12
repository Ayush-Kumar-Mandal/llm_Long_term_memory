# Memory Schema

## Memory Object

```json
{
  "memory_id": "mem_0142",
  "type": "preference",
  "key": "call_time",
  "value": "after 11 AM",
  "source_turn": 1,
  "last_used_turn": 412,
  "confidence": 0.94,
  "embedding": [0.12, 0.87, "..."]
}
```

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `preference` | User likes/dislikes | "Prefers Kannada" |
| `fact` | Stated facts | "Lives in Bangalore" |
| `entity` | Named entities | "Manager is Priya" |
| `constraint` | Hard limitations | "Vegetarian" |
| `commitment` | Promises/tasks | "Report due Friday" |
| `instruction` | Behavior rules | "Always respond formally" |

## Response with Memory Attribution

```json
{
  "response": "I'll schedule the call after 11 AM in Kannada.",
  "active_memories": [
    {
      "memory_id": "mem_0142",
      "content": "User prefers calls after 11 AM",
      "origin_turn": 1,
      "last_used_turn": 937
    }
  ],
  "response_generated": true
}
```
