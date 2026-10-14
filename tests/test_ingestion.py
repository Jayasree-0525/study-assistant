"""Tests for the ingestion pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.chunker import chunk_pages
from ingestion.pdf_extractor import extract_text_from_pdf

TEST_PDF = Path("data/raw/Lecture 2 - External Analysis - Winter 2025 Quercus.pdf")


@pytest.mark.skipif(not TEST_PDF.exists(), reason="Test PDF not found")
def test_pdf_extractor_returns_pages():
    pages = extract_text_from_pdf(str(TEST_PDF))
    assert len(pages) > 0


@pytest.mark.skipif(not TEST_PDF.exists(), reason="Test PDF not found")
def test_pdf_extractor_metadata():
    pages = extract_text_from_pdf(str(TEST_PDF))
    required = {"text", "page_num", "source_file", "content_type", "chunk_hash"}
    for page in pages:
        assert required.issubset(page.keys())


def test_chunker_splits_long_text():
    long_text = "This is a sentence about machine learning. " * 50
    pages = [{"text": long_text, "page_num": 1,
               "source_file": "test.pdf", "content_type": "text", "chunk_hash": "x"}]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunker_preserves_metadata():
    pages = [{"text": "Short text.", "page_num": 5,
               "source_file": "lecture.pdf", "content_type": "text", "chunk_hash": "y"}]
    chunks = chunk_pages(pages)
    assert chunks[0]["page_num"] == 5
    assert chunks[0]["source_file"] == "lecture.pdf"


def test_chunker_no_empty_chunks():
    pages = [{"text": "   \n\n   ", "page_num": 1,
               "source_file": "test.pdf", "content_type": "text", "chunk_hash": "z"}]
    chunks = chunk_pages(pages)
    assert len(chunks) == 0


def test_chunk_hashes_unique():
    pages = [{"text": f"Sentence {i}. " * 30, "page_num": i,
               "source_file": "test.pdf", "content_type": "text", "chunk_hash": f"h{i}"}
             for i in range(5)]
    chunks = chunk_pages(pages)
    hashes = [c["chunk_hash"] for c in chunks]
    assert len(hashes) == len(set(hashes))