"""Table extraction from PDFs using pdfplumber."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pdfplumber


def _table_to_markdown(table: list[list]) -> str:
    if not table or not table[0]:
        return ""
    header = table[0]
    rows = table[1:]
    md = "| " + " | ".join(str(c or "") for c in header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
    return md


def setup_sqlite_db(db_path: str = "data/tables.db") -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extracted_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            page_num INTEGER NOT NULL,
            table_index INTEGER NOT NULL,
            table_markdown TEXT NOT NULL,
            table_json TEXT NOT NULL,
            chunk_hash TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def extract_tables_from_pdf(
    pdf_path: str,
    db_path: str = "data/tables.db",
) -> list[dict]:
    """Extract tables from a PDF and store in SQLite."""
    pdf_path = Path(pdf_path)
    conn = setup_sqlite_db(db_path)
    results = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                markdown = _table_to_markdown(table)
                chunk_hash = hashlib.md5(f"{pdf_path.name}_table_{page_num}_{table_idx}".encode()).hexdigest()
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO extracted_tables
                           (source_file, page_num, table_index,
                            table_markdown, table_json, chunk_hash)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            pdf_path.name,
                            page_num,
                            table_idx,
                            markdown,
                            json.dumps(table),
                            chunk_hash,
                        ),
                    )
                    conn.commit()
                except sqlite3.Error:
                    continue
                results.append(
                    {
                        "text": f"Table on page {page_num}:\n{markdown}",
                        "page_num": page_num,
                        "source_file": pdf_path.name,
                        "content_type": "table",
                        "chunk_hash": chunk_hash,
                    }
                )

    conn.close()
    print(f"Extracted {len(results)} tables from {pdf_path.name}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py path/to/file.pdf")
        sys.exit(1)

    tables = extract_tables_from_pdf(sys.argv[1])
    print(f"Found {len(tables)} tables")
    for t in tables[:2]:
        print(f"\n--- Page {t['page_num']} ---")
        print(t["text"][:400])
