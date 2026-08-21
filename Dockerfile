FROM python:3.11-slim

# set working directory
WORKDIR /app

# install system dependencies needed by PyMuPDF and pdfplumber
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copy dependency files first (Docker layer caching)
COPY pyproject.toml .
COPY requirements-lock.txt .

# install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    langchain \
    langchain-openai \
    langchain-core \
    langchain-text-splitters \
    langgraph \
    openai \
    chromadb \
    pymupdf \
    pdfplumber \
    tavily-python \
    streamlit \
    python-dotenv \
    pydantic \
    typing-extensions

# copy source code
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

# create data directories
RUN mkdir -p data/raw data/processed

# expose Streamlit port
ENV CHROMA_PERSIST_DIR=/tmp/study-assistant/chroma_db
EXPOSE 7860

# set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# run the app
CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
