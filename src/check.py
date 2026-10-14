from collections import Counter

import chromadb
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("study_assistant")
print("count:", col.count())
print("files:", Counter(m["source_file"] for m in col.get()["metadatas"]))

if col.count() > 0:
    r = col.query(
        query_texts=["bargaining power of buyers"],
        n_results=5,
        include=["metadatas", "documents"],
    )
    for m, d in zip(r["metadatas"][0], r["documents"][0]):
        print(m["page_num"], m["content_type"], "|", d[:80].replace("\n", " "))
