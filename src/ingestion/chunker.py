"""Text chunking for RAG pipeline."""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split extracted pages into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []

    for page in pages:
        text = page["text"]
        if not text.strip():
            continue

        chunks = [text] if len(text) <= chunk_size else splitter.split_text(text)

        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            chunk_hash = hashlib.md5(
                f"{page['source_file']}_{page['page_num']}_{chunk_idx}_{chunk_text[:50]}".encode()
            ).hexdigest()

            all_chunks.append(
                {
                    "text": chunk_text.strip(),
                    "page_num": page["page_num"],
                    "source_file": page["source_file"],
                    "content_type": page.get("content_type", "text"),
                    "chunk_index": chunk_idx,
                    "chunk_hash": chunk_hash,
                }
            )

    return all_chunks


if __name__ == "__main__":
    sample = [
        {
            "text": "ML is great. " * 50,
            "page_num": 1,
            "source_file": "test.pdf",
            "content_type": "text",
            "chunk_hash": "x",
        }
    ]
    chunks = chunk_pages(sample)
    print(f"1 long page → {len(chunks)} chunks")
    for c in chunks:
        print(f"  {len(c['text'])} chars")
