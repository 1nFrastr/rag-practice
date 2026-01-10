# RAG Notes - Local Q&A Assistant Edited

A local RAG (Retrieval-Augmented Generation) note Q&A assistant that allows you to upload Markdown/TXT files and ask questions about their content.

## Tech Stack

- Python + uv
- LlamaIndex
- PostgreSQL + pgvector (Docker)
- OpenRouter API (text-embedding-3-small + gpt-4o-mini)
- Gradio Web Interface

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- uv (Python package manager)
- OpenRouter API Key

## Quick Start

### 1. Clone and Setup

```bash
cd rag-notes

# Copy environment template
cp .env.example .env

# Edit .env and add your OpenRouter API key
# OPENROUTER_API_KEY=your_key_here
```

### 2. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

Wait for the database to be ready (check with `docker compose logs -f`).

### 3. Install Dependencies

```bash
uv sync
```

### 4. Run the Application

```bash
uv run python -m src.main
```

The application will be available at http://localhost:7860

## Usage

1. **Upload Documents**: Go to the "Upload Documents" tab, select your `.md` or `.txt` files, and click "Index Documents"

2. **Ask Questions**: Go to the "Ask Questions" tab, enter your question, and click "Ask"

The system will return an answer based on your uploaded documents, along with the relevant source passages.

## Project Structure

```
rag-notes/
├── docker-compose.yml      # PostgreSQL + pgvector
├── pyproject.toml          # Python dependencies
├── .env.example            # Environment template
├── README.md
└── src/
    ├── __init__.py
    ├── main.py             # Application entry point
    ├── config.py           # Configuration management
    ├── indexer.py          # Document indexing
    ├── query_engine.py     # Query processing
    └── app.py              # Gradio interface
```

## Configuration

Environment variables (`.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| OPENROUTER_API_KEY | Your OpenRouter API key | (required) |
| POSTGRES_HOST | PostgreSQL host | localhost |
| POSTGRES_PORT | PostgreSQL port | 5432 |
| POSTGRES_USER | PostgreSQL user | postgres |
| POSTGRES_PASSWORD | PostgreSQL password | postgres |
| POSTGRES_DB | Database name | rag_notes |

## Stopping the Application

```bash
# Stop the web app: Ctrl+C

# Stop PostgreSQL
docker compose down

# Stop and remove data
docker compose down -v
```
