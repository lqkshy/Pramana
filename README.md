# Pramana

Pramana is an AI-powered fact-checking system that decomposes input text into atomic claims, retrieves supporting or refuting evidence for each claim, verifies every claim against that evidence with a language model, and aggregates the per-claim results into a single, explainable verdict.

## Architecture

Pramana is composed of a backend API, a background worker, and support services.

- **API** (`backend/app`): a FastAPI service exposing claim-submission and status endpoints.
- **Pipeline** (`backend/app/pipeline`): orchestrates the `extract → retrieve → verify → aggregate` sequence for each submission.
- **Services** (`backend/app/services`): `ClaimExtractor`, `Retriever`, `Verifier`, and `Aggregator` encapsulate the individual pipeline stages.
- **Models** (`backend/app/models`): Pydantic/SQLAlchemy schemas for `Claim`, `Evidence`, and `Verdict`.
- **Evaluation** (`evaluation`): metrics and a CLI to score verdict quality against a labelled dataset.

## Stack

- Python 3.11+
- FastAPI + Uvicorn (web framework / ASGI server)
- Pydantic + pydantic-settings (configuration and schemas)
- OpenAI Python SDK + httpx (LLM and HTTP access)
- PostgreSQL + Redis (state and queue, via Docker)
- pytest + pytest-asyncio (tests)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in OPENAI_API_KEY
```

Run the whole stack with Docker:

```bash
docker compose up --build
```

Or run the API locally:

```bash
uvicorn backend.app.main:app --reload
```

## Tests

```bash
pytest
```

## Evaluation

```bash
python -m evaluation.run_eval --dataset evaluation/datasets/<dataset>.jsonl
```
