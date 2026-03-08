# AI Agent Evaluation Pipeline

An automated pipeline for continuously evaluating, monitoring, and improving AI agents in production.

## Hosted Demo

| Service | URL |
|---------|-----|
| API (Vercel) | https://ai-eval-pipeline.vercel.app |
| API Docs (Swagger) | https://ai-eval-pipeline.vercel.app/docs |
| GitHub | https://github.com/Ishanuj99/ai-eval-pipeline |

## Architecture Overview

```
Conversations (HTTP POST)
       │
       ▼
  FastAPI API  ──► PostgreSQL
       │
       ▼ (async dispatch)
  Redis Queue
       │
       ▼
  Celery Workers
  ┌────────────────────────────┐
  │  HeuristicEvaluator        │  Fast rule-based checks (no LLM)
  │  ToolCallEvaluator         │  Tool selection + parameter grounding
  │  LLMJudgeEvaluator         │  Gemini-as-judge for quality/helpfulness
  │  CoherenceEvaluator        │  Multi-turn context maintenance
  └────────────────────────────┘
       │
       ▼
  Aggregated Scores → PostgreSQL
       │
       ├──► SuggestionService    (prompt/tool improvement suggestions via LLM)
       └──► MetaEvalService      (evaluator calibration vs human annotations)

  Celery Beat
  ├── Every 30 min: auto_generate_suggestions
  └── Every 2 hrs:  compute_meta_eval_metrics

  Streamlit UI ──► FastAPI
```

### Key Design Decisions

**Why Celery + Redis over Kafka?**
Kafka would be the right choice at true production scale (10k+ msg/s). For this prototype, Celery with Redis provides the same async worker pattern with a much simpler operational footprint. Swapping to Kafka requires only changing the broker URL and adding a consumer group abstraction.

**Why modular evaluators with a base class?**
Each evaluator (`HeuristicEvaluator`, `LLMJudgeEvaluator`, etc.) implements a single `evaluate(conversation) -> dict` interface. New evaluators can be added without touching any other code. The `safe_evaluate()` wrapper ensures one failing evaluator never crashes the whole pipeline.

**Why store raw JSON turns in PostgreSQL?**
Conversation turns are append-only and schema-flexible (tool calls vary by agent). JSONB gives us flexibility without a full document store. Evaluation scores are in structured columns for easy aggregation and filtering.

**Self-Updating Flywheel:**
1. Worker evaluates → stores scores + issues
2. Beat job clusters low-scored conversations
3. Gemini analyzes failure patterns → generates suggestions
4. Human reviews/applies suggestions → improves agent
5. Meta-eval tracks if evaluator scores align with human labels
6. Diverging evaluators get flagged for recalibration

---

## Setup Instructions

### Prerequisites
- Docker & Docker Compose
- A Google Gemini API key (free tier at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone and configure

```bash
git clone https://github.com/Ishanuj99/ai-eval-pipeline
cd ai-eval-pipeline
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 2. Start the stack

```bash
docker compose up --build
```

Services:
| Service | URL |
|---------|-----|
| FastAPI  | http://localhost:8000 |
| Swagger  | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

### 3. Seed demo data

```bash
pip install requests
python scripts/seed_demo.py
```

Wait ~10 seconds for the workers to process, then open the dashboard.

---

## API Documentation

Full Swagger UI at `/docs`. Key endpoints:

### Conversations
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/conversations/` | Ingest single conversation (queues evaluation) |
| `POST` | `/conversations/batch` | Ingest multiple conversations |
| `GET`  | `/conversations/` | List with filters (status, agent_version) |
| `GET`  | `/conversations/{id}` | Get full conversation |

### Evaluations
| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/evaluations/` | List evaluations (filterable by score range) |
| `GET`  | `/evaluations/{id}` | Get single evaluation |
| `GET`  | `/evaluations/by-conversation/{id}` | All evals for a conversation |
| `GET`  | `/evaluations/stats/summary` | Dashboard statistics |
| `POST` | `/evaluations/retry/{conv_id}` | Re-queue failed evaluation |

### Feedback & Annotations
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/feedback/annotations` | Submit human annotation |
| `GET`  | `/feedback/annotations/{conv_id}` | List annotations |
| `GET`  | `/feedback/agreement/{conv_id}/{type}` | Annotator agreement (Kappa) |
| `POST` | `/feedback/meta-eval/compute` | Trigger evaluator calibration |
| `GET`  | `/feedback/meta-eval/metrics` | Evaluator precision/recall/correlation |
| `GET`  | `/feedback/meta-eval/drift` | Flag drifting evaluators |

### Suggestions
| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/suggestions/` | List improvement suggestions |
| `POST` | `/suggestions/generate/sync` | Generate suggestions immediately |
| `PATCH`| `/suggestions/{id}/status` | Mark as applied/rejected |

---

## Evaluation Scoring

Each conversation gets scored across 4 dimensions, combined into an overall score:

| Dimension | Weight | Evaluator |
|-----------|--------|-----------|
| Response Quality | 35% | LLM-as-Judge (Gemini 1.5 Flash) |
| Tool Accuracy | 30% | ToolCallEvaluator (heuristic + grounding) |
| Coherence | 20% | CoherenceEvaluator (Gemini 1.5 Flash) |
| Heuristic | 15% | HeuristicEvaluator (rules) |

### Tool Call Evaluation
- **Selection accuracy**: Was a tool called when the user's intent required one?
- **Parameter accuracy**: Are parameter values traceable to the conversation text?
- **Execution success**: Did the tool return `status: success`?
- **Hallucination**: Were any parameter values fabricated (not grounded in conversation)?

### Annotator Disagreement
- 2 annotators → Cohen's Kappa
- 3+ annotators → Fleiss' Kappa
- Kappa < 0.6 → flagged for human review / tiebreaker

---

## Scaling Strategy

| Scale | Changes Needed |
|-------|---------------|
| 10x (10k conv/min) | Add Celery worker replicas; enable PostgreSQL connection pooling (PgBouncer) |
| 100x (100k conv/min) | Replace Redis with Kafka for ingestion; partition by `agent_version`; use read replicas |
| Beyond | Shard PostgreSQL or migrate evaluation scores to a time-series store (TimescaleDB); cache LLM judge responses for identical turns |

**LLM cost optimization**: At scale, gate LLM evaluators (Gemini judge + coherence) to only run on sampled conversations (e.g., 10% random + 100% of low-heuristic-score). Heuristic and tool evaluators run on 100% at negligible cost.

---

## Trade-offs

| Decision | Optimized For | Trade-off |
|----------|--------------|-----------|
| Celery + Redis | Developer velocity, simple ops | Less throughput than Kafka |
| Gemini 1.5 Flash as judge | Cost + speed (free tier) | Less nuanced than paid models |
| JSON turns in PostgreSQL | Flexibility, no migration pain | Harder to query nested fields |
| Heuristic grounding for tool params | No LLM needed | False positives on complex extractions |
| Synchronous DB writes in workers | Simplicity | Slower than async writes under heavy load |

---

## What I'd Do With More Time

1. **Streaming ingestion**: Add a Kafka-compatible producer endpoint for true high-throughput streaming
2. **Evaluator versioning**: Track which evaluator version produced each score, enable A/B comparing evaluator versions
3. **Regression detection**: Compare scores by `agent_version`; auto-alert on statistically significant drops
4. **Prompt diff tracking**: Store prompt versions alongside evaluations; correlate prompt changes with score changes
5. **Richer coherence eval**: Use sentence embeddings (sentence-transformers) for semantic similarity across turns, not just LLM calls
6. **Auth**: Add API key authentication for multi-tenant use
