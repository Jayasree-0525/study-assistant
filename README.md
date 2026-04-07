# study-assistant
 Multi-modal RAG study assistant with agentic orchestration

# Study Assistant — Multi-modal RAG with Agentic Orchestration

A production-grade AI study assistant that ingests lecture notes, slides,
and handwritten notes, then answers questions and generates quizzes using
a multi-modal RAG pipeline with agentic tool orchestration.

## Live Demo
[Link to Hugging Face Space — add when deployed]

## Architecture
[Architecture diagram — add in Phase 4]

## Features
- Upload PDFs, images, and text files
- Semantic search across all uploaded content
- SQL queries over extracted tables
- Web search fallback via Tavily when content is not in notes
- Quiz generation with structured JSON output
- Session memory for follow-up questions
- RAGAS evaluation scores: [add after Phase 4]

## Tech Stack
| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Vector store | ChromaDB |
| Document parsing | PyMuPDF, pdfplumber |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o |
| Web search | Tavily API |
| Pipeline | Prefect |
| Evaluation | RAGAS |
| Frontend | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Hugging Face Spaces |

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Tavily API key

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/study-assistant.git
cd study-assistant
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env       # Add your API keys to .env
```

### Run locally
```bash
streamlit run src/ui/app.py
```

### Run tests
```bash
pytest tests/ -v
```

## Design Decisions
[Fill this in as you build — explain why you chose each tool]

## Evaluation Results
[RAGAS scores — add after Phase 4]

## Project Status
- [x] Phase 1: Setup and environment
- [ ] Phase 2: Document ingestion pipeline
- [ ] Phase 3: Agentic retrieval system
- [ ] Phase 4: Orchestration and evaluation
- [ ] Phase 5: Deployment
