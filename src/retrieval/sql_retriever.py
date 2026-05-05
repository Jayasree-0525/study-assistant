"""LLM-powered SQL retrieval over extracted tables."""

import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_PATH = "data/tables.db"

SCHEMA = """
Table: extracted_tables
Columns:
  - id (INTEGER): primary key
  - source_file (TEXT): PDF filename
  - page_num (INTEGER): page number in the PDF
  - table_index (INTEGER): index of table on that page
  - table_markdown (TEXT): table content as markdown
  - chunk_hash (TEXT): unique identifier
  - created_at (TIMESTAMP): when it was ingested
"""

# this whole file basically is an upgraded version of the query_tables function in tools.py
# this allows the agent to use the user query and get GPT4o to convert it into a SQL query which can be executed against the SQLite DB & get results back.
# This is more flexible and powerful than the previous hardcoded SQL queries in tools.py.


def _generate_sql(question: str, client: OpenAI) -> str:
    """Use GPT to convert a natural language question to SQL."""
    prompt = f"""You are a SQL expert. Given this database schema:

{SCHEMA}

Write a SQL SELECT query to answer this question:
"{question}"

Rules:
- Only write SELECT statements, never INSERT/UPDATE/DELETE/DROP
- Keep it simple — avoid complex joins
- Return ONLY the SQL query, nothing else
- If you cannot answer with this schema, return: SELECT 'Cannot answer' as result
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```sql\n?|```\n?", "", sql).strip()
    return sql


# this function checks if the generated SQL query is safe to execute by rejecting any queries that contain keywords associated
# with data modification or schema changes.
def _is_safe_sql(sql: str) -> bool:
    """Reject any SQL that could modify data."""
    dangerous = ["insert", "update", "delete", "drop", "alter", "create"]
    sql_lower = sql.lower()
    return not any(word in sql_lower for word in dangerous)


# this function orchestrates everything: generates the SQL, safety checks it, runs it, and formats the results.
def query_tables_with_llm(question: str) -> str:
    """
    Convert a natural language question to SQL and execute it
    against the extracted tables database.

    Args:
        question: Natural language question about table data.

    Returns:
        Query results as a formatted string.
    """
    if not Path(DB_PATH).exists():
        return "No table database found."

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sql = _generate_sql(question, client)

    if not _is_safe_sql(sql):
        return f"Generated SQL was rejected for safety: {sql}"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        conn.close()

        if not rows:
            return "Query returned no results."

        result_lines = [" | ".join(cols)]
        result_lines.append("-" * 40)
        for row in rows[:10]:
            result_lines.append(" | ".join(str(v)[:100] for v in row))

        return f"SQL: {sql}\n\nResults:\n" + "\n".join(result_lines)

    except sqlite3.Error as e:
        return f"SQL error: {e}\nGenerated SQL was: {sql}"


if __name__ == "__main__":
    questions = [
        "How many tables were extracted?",
        "Which pages have tables?",
        "Show me tables from page 5",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(query_tables_with_llm(q))
