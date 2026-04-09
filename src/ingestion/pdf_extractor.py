"""PDF text extraction module."""

import hashlib
from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Returns:
        List of dicts, one per page:
        {
            "text": str,
            "page_num": int,
            "source_file": str,
            "content_type": "text",
            "chunk_hash": str
        }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    doc = fitz.open(str(pdf_path))

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if not text:
            continue

        chunk_hash = hashlib.md5(
            f"{pdf_path.name}_{page_num}_{text[:100]}".encode()
        ).hexdigest()

        pages.append(
            {
                "text": text,
                "page_num": page_num,
                "source_file": pdf_path.name,
                "content_type": "text",
                "chunk_hash": chunk_hash,
            }
        )

    doc.close()
    return pages


if __name__ == "__main__":
    import sys

    results = extract_text_from_pdf(sys.argv[1])
    print(f"Extracted {len(results)} pages")
    for r in results[:2]:
        print(f"\n--- Page {r['page_num']} ---")
        print(r["text"][:300])
