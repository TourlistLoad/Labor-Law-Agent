# Labor-Law-Agent

Agentic RAG assistant for Vietnamese labor law. It answers questions with citations to the exact article and clause, and computes severance and overtime pay with deterministic, tested tools instead of letting the LLM do arithmetic.

> **Status:** work in progress, built over 30 days as a hands-on study of agentic RAG.
> **Not legal advice.** Answers always link to the source text; consult a lawyer for real decisions.

## Planned features

- Hybrid search (BM25 + dense vectors) with reranking over the Labor Code 2019 and its guiding decrees, chunked by Chapter, Article and Clause
- An agent that decides when to retrieve, when to call a calculation tool, and when to ask the user for missing details
- Self-correcting retrieval (Corrective RAG): grades retrieved passages and rewrites the query when they are off-topic
- Multi-agent supervisor (researcher, calculator, drafter) with human approval before drafting letters
- Memory of the user's profile across sessions
- Retriever exposed as an MCP server
- Automated evaluation in CI (retrieval recall, faithfulness, tool-choice accuracy)
- Public deployment on free-tier infrastructure

## Tech stack

| Layer | Choice |
| --- | --- |
| Agent framework | LangChain 1.x, LangGraph 1.x |
| Data validation | Pydantic v2 |
| LLM | Gemini (primary), Groq (fallback) |
| Embeddings and reranking | bge-m3, bge-reranker-v2-m3 |
| Vector store | Qdrant |
| State and memory | Postgres (LangGraph checkpointer and store) |
| API | FastAPI with server-sent events |
| Tracing and evaluation | LangSmith or Langfuse, RAGAS |
| Deployment | Docker on Hugging Face Spaces, GitHub Actions |

## Progress

- [x] Day 01: Pydantic schemas for the severance tool, self-correcting output parser (30/30 tests)
- [ ] Week 1: LLM fundamentals, ingestion, naive RAG baseline
- [ ] Week 2: Advanced retrieval, evaluation, single agent with tools
- [ ] Week 3: LangGraph workflows, persistence, human-in-the-loop, multi-agent, MCP
- [ ] Week 4: FastAPI, guardrails, Docker, CI, deployment

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -v
```

## Repository layout

```text
learning/   daily exercises with self-checking tests
notes/      daily study notes
```

The application code (`app/`, `evals/`) will be added from week 1 onward.
