# Pramana

Research-grade, open-source AI fact-checking pipeline. Paste text → get every checkable claim
with a verdict (**SUPPORTED / CONTRADICTED / CONFLICTING_EVIDENCE / INSUFFICIENT_EVIDENCE**),
a plain-English chain-of-thought, cited sources, and an `evidence_strength` score.
Runs on free tiers only (Groq, Tavily, Supabase).

## Pipeline (9 stages)

| # | Stage | Module | Tooling |
|---|-------|--------|---------|
| 1 | Claim extraction (1 call) | `app/pipeline/claims/extractor.py` | Groq Llama 3.1 8B |
| 2 | Check-worthiness filter | `app/pipeline/check_worthiness.py` | Groq 8B |
| 3 | Claim matching (cache) | `app/pipeline/claim_matching.py` | MiniLM + pgvector |
| 4 | Query generation | `app/pipeline/queries.py` | Groq 8B |
| 5 | Retrieval (live / pre-retrieved) | `app/pipeline/retrieval.py` | Tavily |
| 6 | Source trust scoring | `app/services/source_trust.py` | pure Python |
| 7 | Evidence: scrape → rerank → refine | `app/pipeline/evidence.py`, `reranking.py` | cross-encoder + 1 Groq call |
| 8 | Verification + CoT | `app/pipeline/verification.py` | Groq 8B (dev) / 3.3 70B (prod) |
| 9 | Store + return | `app/models/database.py` | Supabase |

`evidence_strength = avg(quality of cited sources) × min(1, cited_count / 3)`

## Quick start

```bash
git clone https://github.com/lqkshy/Pramana.git
cd Pramana
python -m venv .venv && .venv\Scripts\activate      # Windows (use Python 3.11/3.12)
pip install -r requirements.txt
copy .env.example .env                               # then fill in your keys
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs · Health: http://localhost:8000/health

```bash
curl -X POST http://localhost:8000/verify -H "Content-Type: application/json" \
     -d '{"text": "NASA launched Artemis I on November 16, 2022."}'
```

## Configuration (`.env`)

| Var | Meaning |
|-----|---------|
| `LLM_PROVIDER` | `groq` (default) · `gemini` · `anthropic` · `openai` · `ollama` |
| `GROQ_API_KEY` / `TAVILY_API_KEY` | free keys from console.groq.com / tavily.com |
| `DATABASE_URL` | Supabase Postgres URL (optional — falls back to in-memory cache) |
| `DEV_MODE` | `true` = verification on 8B (14,400 req/day). `false` = 3.3 70B for benchmarks/demo |
| `USE_LIVE_SEARCH` | `false` = pre-retrieved evidence (benchmarks). The API server forces `true` |

## Tests

```bash
pytest -v          # runs offline; live-API and embedding tests auto-skip without keys/models
```

## Layout

```
app/
  main.py                FastAPI app (/health, /verify)
  models/                schemas.py (Pydantic), database.py (SQLAlchemy + pgvector)
  pipeline/              stages 1-5, 7, 8 (+ claims/extractor.py)
  services/              llm_client.py, source_trust.py, logger.py
evaluation/              benchmark loaders + metrics (AVeriTeC, SciFact)
tests/                   pytest suite
```

## License
MIT
