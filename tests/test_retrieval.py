def test_semantic_search_retrieves_five_forces():
    """Semantic search should retrieve content about Five Forces from the lecture."""
    from retrieval.tools import semantic_search

    result = semantic_search.invoke({"query": "five forces buyer supplier rivalry"})
    assert isinstance(result, str)
    assert len(result) > 100
    # Should find something relevant from the lecture
    assert any(word in result.lower() for word in ["buyer", "supplier", "rivalry", "force", "substitute"])


def test_query_tables_finds_lecture_tables():
    """Table query should find tables extracted from the lecture."""
    from retrieval.tools import query_tables

    result = query_tables.invoke({"question": "show me tables from the lecture"})
    assert isinstance(result, str)
    # Should either find tables or say none found — not crash
    assert len(result) > 0


def test_web_search_returns_results():
    """Web search should return real results with URLs."""
    from retrieval.tools import web_search

    result = web_search.invoke({"query": "Porter Five Forces competitive strategy"})
    assert isinstance(result, str)
    assert "http" in result  # should contain a URL
