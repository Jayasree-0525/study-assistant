---
title: Study Assistant
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Study Assistant — Multi-modal RAG with Agentic Orchestration

A production-grade AI study assistant that ingests lecture notes and PDFs,
then answers questions and retrieves cited content using a multi-modal RAG
pipeline with agentic tool orchestration built on LangGraph.

## Live Demo
[🚀 Try it on Hugging Face Spaces](https://huggingface.co/spaces/jayasree-s/study-assistant)

## What it does
Upload any lecture PDF and ask questions about it. The agent:
- Searches your notes semantically using ChromaDB vector search
- Queries structured table data extracted from slides
- Falls back to web search when notes don't contain the answer
- Remembers conversation history within a session
- Cites the exact page number for every answer

## Architecture
```
User uploads PDF
      ↓
Ingestion pipeline (PyMuPDF + pdfplumber + OpenAI embeddings)
      ↓
ChromaDB vector store (138 chunks per lecture)
      ↓
LangGraph ReAct agent (GPT-4o-mini + 3 tools)
      ↓
Streamlit frontend → cited answer with page numbers
```

## Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Agent orchestration | LangGraph |
| LLM | GPT-4o-mini |
| Vector store | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| Document parsing | PyMuPDF + pdfplumber |
| Web search | Tavily API |
| Pipeline orchestration | Prefect |
| CI/CD | GitHub Actions |
| Deployment | Docker + Hugging Face Spaces |

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Tavily API key
- Docker (for containerized deployment)

### Local installation
```bash
git clone https://github.com/Jayasree-0525/study-assistant.git
cd study-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
streamlit run src/ui/app.py
```

### Docker
```bash
docker build -t study-assistant .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your-key \
  -e TAVILY_API_KEY=your-key \
  study-assistant
```

## How the agent works
The agent uses a ReAct loop — it reasons about which tool to call,
executes the tool, reads the result, and decides whether to call
another tool or synthesise a final answer. Three tools are available:

1. **semantic_search** — embeds the query and searches ChromaDB
2. **query_tables** — queries structured data extracted from slides
3. **web_search** — Tavily API fallback for external context

## Project Status
- [x] Phase 1: Setup and environment
- [x] Phase 2: Multi-modal document ingestion pipeline
- [x] Phase 3: LangGraph agent with tools and memory
- [x] Phase 4: Streamlit frontend
- [x] Phase 5: Docker and Hugging Face deployment
