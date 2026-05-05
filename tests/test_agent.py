"""Tests for the agent tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_semantic_search_returns_string():
    """Semantic search tool should always return a string."""
    from retrieval.tools import semantic_search

    result = semantic_search.invoke({"query": "Five Forces model"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_semantic_search_no_results():
    """Semantic search should handle queries with no matches gracefully."""
    from retrieval.tools import semantic_search

    result = semantic_search.invoke({"query": "xyzzy quantum blockchain unicorn"})
    assert isinstance(result, str)


def test_query_tables_returns_string():
    """Table query tool should always return a string."""
    from retrieval.tools import query_tables

    result = query_tables.invoke({"question": "what tables exist?"})
    assert isinstance(result, str)


def test_web_search_returns_string():
    """Web search tool should return a string."""
    from retrieval.tools import web_search

    result = web_search.invoke({"query": "Porter Five Forces model"})
    assert isinstance(result, str)


def test_semantic_search_finds_relevant_content():
    """Semantic search should find content specific to the lecture."""
    from retrieval.tools import semantic_search

    result = semantic_search.invoke({"query": "buyer power bargaining"})
    assert isinstance(result, str)
    assert "lecture.pdf" in result


def test_all_tools_importable():
    """ALL_TOOLS list should contain exactly three tools."""
    from retrieval.tools import ALL_TOOLS

    assert len(ALL_TOOLS) == 3
