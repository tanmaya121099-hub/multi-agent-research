# Architecture Deep Dive

## Agent Roles and Separation of Concerns

Each agent has one job and knows nothing about the others. They share state through Redis/Postgres, not through direct calls.

```
Planner    → owns: task decomposition, ordering
Researcher → owns: web search, fact extraction, citation
Critic     → owns: quality evaluation, gap detection
Writer     → owns: synthesis, formatting, coherent voice
```

## LangGraph State Machine

```
START
  │
  ▼
[planner]
  │ produces ResearchPlan with ordered SubTask list
  ▼
[researcher] ◄──────────────────────────────────────┐
  │ processes subtask[current_index]                │
  ├── more subtasks? → loop back                   │
  └── all done? → [critic]                         │
                       │                            │
                 approved?                          │
                    │                               │
              ┌─────┴──────┐                       │
              │ no         │ yes                    │
              │            ▼                        │
              │        [writer]                     │
              │            │                        │
              │            ▼                        │
              │           END                       │
              │                                     │
        iterations < MAX?                           │
              ├── yes → add missing_topics to query ┘
              └── no  → [writer] (degrade gracefully)
```

## Memory Hierarchy

```
┌───────────────────────────────────────────────────┐
│                  Memory Tiers                     │
│                                                   │
│  Hot (Redis 1hr)    → current subtask context     │
│  Working (Redis 24h) → all facts + sources        │
│  Cold (Postgres)    → session history, reports    │
│  Semantic (pgvector) → embedding search           │
│                                                   │
│  Agents write to Redis; API persists to Postgres  │
│  Semantic layer lets future sessions reuse facts  │
└───────────────────────────────────────────────────┘
```

## Typed Message Contracts

All inter-agent data flows through Pydantic models, never raw dicts:

```python
SubTask(title, description, status, assigned_to)
ResearchPlan(query, subtasks, session_id)
Fact(text, source_url, source_title, confidence)
CritiqueResult(approved, issues, missing_topics, overall_score)
AgentMessage(sender, receiver, action, payload, timestamp, session_id)
```

## Critic Circuit Breaker

```
critic_iterations < MAX_CRITIC_RETRIES (default: 2)
  → route back to researcher with missing_topics appended to plan

critic_iterations >= MAX_CRITIC_RETRIES
  → route to writer regardless (graceful degradation)
  → log warning: "critic.max_retries_reached"
```

This prevents research from stalling indefinitely on topics with no good web coverage.

## Agent Model Selection Rationale

| Agent | Model | Reasoning |
|-------|-------|-----------|
| Planner | GPT-4o-mini | Simple decomposition, low complexity |
| Researcher | GPT-4o-mini | Fact extraction from given context — no reasoning |
| Critic | GPT-4o-mini | Structured evaluation with JSON schema |
| Writer | Claude Sonnet 4.6 | Long-form synthesis, nuance, consistent voice |

Cost breakdown (per research query, ~4 subtasks):
- Planner: 1 call × ~500 tokens ≈ $0.00015
- Researcher: 4 calls × ~2000 tokens ≈ $0.0012
- Critic: 1–2 calls × ~800 tokens ≈ $0.0005
- Writer: 1 call × ~3000 tokens ≈ $0.015
- **Total: ~$0.02–0.06 per query**

## Failure Handling

| Failure | Behavior |
|---------|---------|
| Redis down | Memory writes silently fail; research still completes |
| Postgres down | Session not persisted; response still returned |
| Search API failure | `tenacity` retries 3×; empty results returned |
| Agent LLM timeout | `tenacity` retries 3×; then raises |
| Scraper blocked | Returns empty string; research continues |
| Max critic retries | Degrades to writer with available facts |
