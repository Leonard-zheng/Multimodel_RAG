# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entry; runs the full MultiRAG pipeline.
- `src/partition.py`: PDF parsing via Unstructured; extracts text, tables, images.
- `src/summaries.py`: text/image summarization chains using LangChain and `config/prompt.yml`.
- `src/rag_pipeline.py`: orchestrates retrieval + generation; builds the RAG chains.
- `src/vector_store.py`: Chroma persistence and `MultiVectorRetriever` wiring; docstore persisted to `docstore.pkl`.
- `src/llm_manager.py`: cached LLM/embeddings (Google GenAI, optional Ollama).
- `src/config.py`: lightweight dataclass settings with sane defaults; environment variables are only used for provider SDKs (e.g., `GOOGLE_API_KEY`).
- `content/`: sample inputs; `chroma_db/`: local vector store; `cache/`: summaries cache.
- `requirements.txt`: runtime deps; dev tools commented out.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Configure env: `cp .env.example .env` and set `GOOGLE_API_KEY` (required for Google GenAI). Other `.env` keys shown are optional and may not be read by `src/config.py` (which uses internal defaults).
- Run locally: `python main.py`
- Optional tools: `pip install pytest black mypy`
  - Tests: `pytest -q`
  - Format: `black .`
  - Type check: `mypy .`

## Coding Style & Naming Conventions
- Follow PEP 8; 4-space indentation; UTF-8 files.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, constants `UPPER_SNAKE_CASE`.
- Type hints for public functions; concise docstrings aligned with existing modules.
- Prefer small, single-purpose functions; handle errors via `utils.handle_errors`.

## Testing Guidelines
- Framework: pytest (optional dev dependency). Place tests under `tests/` with `test_*.py`.
- Focus: `partition()`, summarizers in `summaries.py`, `RAG.call()` happy/error paths.
- Mock external I/O and LLM calls (`llm_manager.get_llm()`) for determinism.
- Quick run: `pytest -q`; aim to cover critical branches.

## Commit & Pull Request Guidelines
- Commits: prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`). Keep messages imperative and scoped.
- PRs: include summary, rationale, test plan/outputs, any config changes (env vars), and impact on `chroma_db/` (note if reindex needed).
- Link related issues; add screenshots or logs when helpful.

## Security & Configuration Tips
- Do not commit secrets; keep `.env` local. Required: `GOOGLE_API_KEY`.
- Logging level/file can be controlled by code defaults in `src/utils.py` and `src/config.py`. `.env` keys for logging are optional and not strictly used by settings.
- `chroma_db/` persists vectors for local runs; delete the folder to rebuild the index when schemas or embeddings change.
- Docstore for original content is now stored in `docstore.pkl` (pickle). Legacy `docstore.json` may exist but was also pickle-encoded; both should be treated as ephemeral and ignored by Git.
