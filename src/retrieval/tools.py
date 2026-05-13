"""The three retrieval tools available to the LangGraph agent."""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

DB_PATH = "data/tables.db"


@tool
def semantic_search(query: str, n_results: int = 5) -> str:
    """
    Search the student's uploaded notes for content
    semantically similar to the query.

    Use this tool first for any question. Returns the most
    relevant chunks from ChromaDB with their source and page number.

    Args:
        query: The question or search text.
        n_results: Number of results to return (default 5).

    Returns:
        Formatted string of relevant chunks with citations.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.vector_store import query_chroma

    results = query_chroma(query, n_results=n_results)

    if not results:
        return "No relevant content found in uploaded notes."

    output = []
    for i, r in enumerate(results, 1):
        output.append(f"[Result {i} | {r['source_file']} p.{r['page_num']} | score: {r['similarity_score']}]\n{r['text']}")

    return "\n\n---\n\n".join(output)


@tool
def query_tables(question: str) -> str:
    """
    Answer questions about structured/tabular data extracted
    from the student's notes. Use this when the question
    involves data in a table, comparison, or structured list.

    Args:
        question: Natural language question about table data.

    Returns:
        Relevant table content as markdown, or a message if
        no tables are found.
    """
    if not Path(DB_PATH).exists():
        return "No table database found. Tables may not have been extracted."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        SELECT source_file, page_num, table_markdown
        FROM extracted_tables
        ORDER BY page_num
        LIMIT 10
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No tables found in the uploaded documents."

    question_lower = question.lower()
    relevant = []
    for source, page, markdown in rows:
        if any(word in markdown.lower() for word in question_lower.split()):
            relevant.append(f"[Table from {source} p.{page}]\n{markdown}")

    if not relevant:
        all_tables = [f"[Table from {s} p.{p}]\n{m}" for s, p, m in rows[:3]]
        return "No exact table match found. Here are available tables:\n\n" + "\n\n---\n\n".join(all_tables)

    return "\n\n---\n\n".join(relevant[:3])


@tool
def web_search(query: str) -> str:
    """
    Search the web for information NOT found in the student's
    uploaded notes. Use this as a fallback when semantic_search
    returns low-confidence results or the topic needs
    external context.

    Args:
        query: Search query for the web.

    Returns:
        Top web search results with titles and snippets.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key not configured. Web search unavailable."

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=3,
    )

    results = response.get("results", [])
    if not results:
        return "No web results found."

    output = []
    for r in results:
        output.append(f"[{r.get('title', 'No title')}]\nURL: {r.get('url', '')}\n{r.get('content', '')[:300]}")

    return "\n\n---\n\n".join(output)


ALL_TOOLS = [semantic_search, query_tables, web_search]
