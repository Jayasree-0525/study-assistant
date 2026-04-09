"""ChromaDB vector store for storing and retrieving embedded chunks."""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "study_assistant"


def get_chroma_client() -> chromadb.PersistentClient:
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection(client: chromadb.PersistentClient):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks_to_chroma(chunks: list[dict], batch_size: int = 50) -> int:
    """Embed and store chunks. Skips duplicates using chunk_hash as ID."""
    client = get_chroma_client()
    collection = get_collection(client)

    existing_ids = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_hash"] not in existing_ids]

    if not new_chunks:
        print("No new chunks — all already in ChromaDB.")
        return 0

    print(f"Adding {len(new_chunks)} new chunks...")
    added = 0
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_hash"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "page_num": c["page_num"],
                    "source_file": c["source_file"],
                    "content_type": c["content_type"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )
        added += len(batch)
        print(f"  Batch {i // batch_size + 1} done — {added} total")

    print(f"ChromaDB total: {collection.count()} chunks")
    return added


def query_chroma(
    query: str,
    n_results: int = 5,
    source_file: str | None = None,
) -> list[dict]:
    """Search ChromaDB for chunks most similar to the query."""
    client = get_chroma_client()
    collection = get_collection(client)

    where = {"source_file": source_file} if source_file else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "text": doc,
                "source_file": meta["source_file"],
                "page_num": meta["page_num"],
                "content_type": meta["content_type"],
                "similarity_score": round(1 - dist, 3),
            }
        )
    return output
