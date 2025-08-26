# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multimodal RAG (Retrieval-Augmented Generation) system that processes PDF documents containing text, tables, and images. The system extracts and summarizes different content types, stores them in a vector database, and enables querying through a conversational interface.

## Architecture

### Core Components

- `src/partition.py`: Document processing with error handling and file validation
- `src/summaries.py`: Content summarization using cached LLM instances for better performance
- `src/vector_store.py`: Multi-vector retrieval system using Chroma DB with Google embeddings; docstore persisted as `docstore.pkl`
- `src/rag_pipeline.py`: Optimized RAG pipeline with lazy chain building and error handling
- `src/llm_manager.py`: Singleton LLM manager to avoid costly instance recreation
- `src/utils.py`: Centralized error handling, logging, and validation utilities
- `src/config.py`: Dataclass-based configuration with internal defaults (no Pydantic). `.env` is used for provider API keys (e.g., `GOOGLE_API_KEY`).
- `main.py`: Entry point with logging and error handling

### Data Flow

1. PDF document → `partition()` → separate text/tables/images
2. Content → `summarize()` functions → summaries for each content type
3. Original content + summaries → `DocumentManager.add_documents()` → vector store + in-memory store (persisted to `docstore.pkl`)
4. Query → `RAG.call()` → retrieval + context building + LLM response

## Key Dependencies

- LangChain core + integrations (Chroma, Google GenAI, Ollama)
- Google Generative AI: LLM (gemini-2.5-flash-lite) and embeddings (gemini-embedding-001)
- Chroma: vector database persisted to `./chroma_db/`
- Unstructured: PDF processing with table structure inference and image extraction
- python-dotenv: loads environment variables from `.env` for SDKs

## Running the System

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Environment configuration:
```bash
cp .env.example .env
# Edit .env file and set GOOGLE_API_KEY
```

3. Run the system:
```bash
python main.py
```

### Testing and Development
- Use notebooks in `notebooks/` for quick experiments
- Monitor logs for debugging and performance insights
- Errors are caught and logged with context via decorators in `utils.py`

## Configuration

- `config/prompt.yml`: Prompts for text/table and image summarization
- Content directory: Place PDFs in `./content/` (default: `attention-is-all-you-need.pdf`)
- Vector DB: persisted in `./chroma_db/`
- Docstore: original content persisted in `docstore.pkl` (pickle). A legacy `docstore.json` may exist but is also pickled; treated as ephemeral.

## Recent Improvements (2024)

### Performance Optimizations
- LLM Instance Management: Singleton pattern prevents costly LLM recreation
- Lazy Chain Building: RAG chains built once on first use
- Efficient Caching: LLM and embedding instances cached with thread safety

### Error Handling & Reliability
- Comprehensive error handling: decorator-based logging and exception translation
- File validation before processing
- Structured logging with consistent format
- Custom exceptions per domain (processing, vector store, RAG)

### Configuration Management
- Dataclass-based config with in-code defaults in `src/config.py`
- `.env` used for provider API keys; settings like model/temperature are configured in code

### Code Quality
- Type annotations on key functions
- Pinned dependencies in `requirements.txt`
- Clean separation of concerns

## Important Notes

- Hybrid storage: summaries in vector store for retrieval; original content in an in-memory store persisted to `docstore.pkl`
- Image inputs are handled as base64 for multimodal LLMs
- Chunking strategy: by_title with configurable character limits
- LLM instances are reused across the application for better performance
