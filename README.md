# RoadVision Copilot

An AI-powered traffic analysis system that combines computer vision with LLM-based reasoning to detect German traffic signs, look up their meaning under the StVO (Straßenverkehrs-Ordnung), and generate reports.

> WBS Coding School graduation project (Data Science & AI Bootcamp).

## Overview

RoadVision Copilot processes traffic footage through a CV pipeline (YOLO-based sign detection) and feeds the detected sign classes into an LLM-backed chatbot that:

- Looks up the corresponding StVO regulation for each detected sign
- Answers questions about traffic signs and regulations
- Generates structured reports summarizing findings

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Frontend    │ ───▶ │   Backend (API)   │ ───▶ │  ChromaDB    │
│  (Streamlit) │      │   (FastAPI)       │      │  (RAG store) │
└─────────────┘      └──────────────────┘      └─────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │  CV Pipeline      │
                      │  (YOLO, 43 GTSDB  │
                      │   sign classes)   │
                      └──────────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │  Claude (Haiku +  │
                      │  Sonnet)          │
                      └──────────────────┘
```

**Flow:** CV pipeline detects traffic signs → sign numbers are passed to the backend → backend retrieves StVO meanings via RAG → LLM generates lookups (Claude Haiku) and reports (Claude Sonnet).

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Orchestration:** Docker Compose
- **LLMs:** Claude Haiku (sign lookups), Claude Sonnet (report generation)
- **RAG:** LlamaIndex, ChromaDB (two-collection setup: `signs_json` + `stvo_full`), custom BM25 retriever (`rank_bm25`) for hybrid search
- **Storage:** SQLite
- **Linting/Formatting:** Ruff (enforced via pre-commit)

## Project Structure

```
roadvision_copilot/
├── backend/
│   ├── data/
│   ├── llm/
│   ├── routes/
│   ├── Dockerfile
│   ├── main.py
│   ├── models.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── data/
├── .env
├── .gitignore
├── docker-compose.yml
├── ruff.toml
├── .pre-commit-config.yaml
└── README.md
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- An Anthropic API key

### Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:MarvinAtorf/roadvision_copilot.git
   cd roadvision_copilot
   ```

2. Create a `.env` file in the project root with the required variables (see `.env` section below).

3. Start the stack:
   ```bash
   docker compose up -d --build
   ```

4. Access the services:
   - Backend API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - API docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Frontend (Streamlit): [http://127.0.0.1:8501](http://127.0.0.1:8501)

### Environment Variables

```
ANTHROPIC_API_KEY=your_key_here
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Development

### Linting & Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, enforced automatically via pre-commit hooks.

```bash
ruff check .          # Lint
ruff check . --fix    # Lint with auto-fix
ruff format .         # Format
```

Set up pre-commit hooks once:
```bash
pre-commit install
```

## Roadmap

- [ ] Migrate `/chat` endpoint conditional logic to a proper LangGraph `conditional_edge` implementation

## License

TBD