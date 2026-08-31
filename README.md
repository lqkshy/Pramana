# Pramana

An AI-powered fact-checking API that extracts claims from text, searches for evidence, and verifies accuracy — built for developers who need reliable fact-checking in their applications.

---

## Tech Stack

- **FastAPI** — high-performance async API framework
- **Groq (Llama 3.1)** — fast LLM inference for claim extraction and reasoning
- **Tavily Search** — real-time web search for evidence retrieval
- **Supabase (PostgreSQL)** — persistent storage for claims, evidence, and verdicts
- **Next.js** — frontend dashboard (coming soon)

---

## Quick Start

```bash
git clone https://github.com/YOURNAME/pramana.git
cd pramana
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
uvicorn main:app --reload
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

---

## Project Structure

| Folder | Purpose |
|--------|---------|
| `app/pipeline/claims/` | Claim extraction, disambiguation, and decomposition |
| `app/pipeline/` | Retrieval and verification pipeline stages |
| `app/services/` | LLM client, logging, and shared utilities |
| `app/models/` | Pydantic schemas and database models |
| `app/api/routes/` | FastAPI endpoint definitions |
| `tests/` | pytest test suite |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | LLM provider: `groq`, `gemini`, `anthropic`, `openai`, `ollama` (default: `groq`) |
| `GROQ_API_KEY` | Groq API key for Llama 3.1 inference |
| `GEMINI_API_KEY` | Google Gemini API key (if using gemini provider) |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using anthropic provider) |
| `OPENAI_API_KEY` | OpenAI API key (if using openai provider) |
| `TAVILY_API_KEY` | Tavily Search API key for web evidence retrieval |
| `DATABASE_URL` | PostgreSQL connection string (Supabase or local) |
| `DEV_MODE` | `true` or `false` — enables verbose logging and smaller models |
| `USE_LIVE_SEARCH` | `true` or `false` — enable/disable Tavily live search |

---

## Week 1 Status

✅ **Working:**
- Unified async LLM client with rate limiting (Groq, Gemini, Anthropic, OpenAI, Ollama)
- Claim extraction with JSON-structured output (selected claims, disambiguated, decomposed)
- FastAPI server with CORS and structured logging
- Supabase/PostgreSQL connection and schema
- Tavily Search integration for evidence retrieval

🔜 **Coming:**
- Evidence verification and verdict generation
- Next.js frontend dashboard
- Evaluation benchmark (TruthfulQA, FEVER, custom datasets)
- Caching layer and request deduplication

---

## License

MIT