"""Main ingestion pipeline."""

import hashlib
from pathlib import Path

from ingestion.chunker import chunk_pages
from ingestion.pdf_extractor import extract_text_from_pdf
from ingestion.table_extractor import extract_tables_from_pdf
from ingestion.vector_store import add_chunks_to_chroma


def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_document(
    file_path: str,
    extract_images: bool = False,
    db_path: str = "data/tables.db",
) -> dict:
    """Full ingestion pipeline for a single PDF document."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\nIngesting: {file_path.name}")
    print("=" * 50)

    summary = {
        "file": file_path.name,
        "file_hash": _file_hash(str(file_path)),
        "text_pages": 0,
        "image_pages": 0,
        "tables": 0,
        "total_chunks": 0,
        "chunks_added": 0,
    }

    all_chunks = []

    print("1. Extracting text...")
    text_pages = extract_text_from_pdf(str(file_path))
    summary["text_pages"] = len(text_pages)
    text_chunks = chunk_pages(text_pages)
    all_chunks.extend(text_chunks)
    print(f"   {len(text_pages)} pages → {len(text_chunks)} chunks")

    if extract_images:
        print("2. Extracting visuals (GPT-4o)...")
        from ingestion.image_extractor import extract_images_from_pdf

        image_pages = extract_images_from_pdf(str(file_path), max_pages=10)
        summary["image_pages"] = len(image_pages)
        image_chunks = chunk_pages(image_pages)
        all_chunks.extend(image_chunks)
    else:
        print("2. Skipping image extraction (extract_images=False)")

    print("3. Extracting tables...")
    table_data = extract_tables_from_pdf(str(file_path), db_path=db_path)
    summary["tables"] = len(table_data)
    all_chunks.extend(chunk_pages(table_data))

    summary["total_chunks"] = len(all_chunks)
    print(f"4. Embedding {len(all_chunks)} chunks...")
    summary["chunks_added"] = add_chunks_to_chroma(all_chunks)

    print(f"\nDone: {summary}")
    return summary


if __name__ == "__main__":
    import sys

    ingest_document(sys.argv[1], extract_images=False)
