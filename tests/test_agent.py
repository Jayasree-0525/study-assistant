"""Tests for the agent tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_semantic_search_returns_string():
    from retrieval.tools import semantic_search

    result = semantic_search.invoke({"query": "Five Forces model"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_semantic_search_no_results():
    from retrieval.tools import semantic_search

    result = semantic_search.invoke({"query": "xyzzy quantum blockchain unicorn"})
    assert isinstance(result, str)


def test_query_tables_returns_string():
    from retrieval.tools import query_tables

    result = query_tables.invoke({"question": "what tables exist?"})
    assert isinstance(result, str)


def test_web_search_returns_string():
    from retrieval.tools import web_search

    result = web_search.invoke({"query": "Porter Five Forces model"})
    assert isinstance(result, str)


def test_semantic_search_finds_relevant_content():
    from retrieval.tools import semantic_search

    result = semantic_search.invoke({"query": "buyer power bargaining"})
    assert isinstance(result, str)
    assert "Lecture" in result or "lecture" in result


def test_all_tools_importable():
    from retrieval.tools import ALL_TOOLS

    assert len(ALL_TOOLS) == 3
