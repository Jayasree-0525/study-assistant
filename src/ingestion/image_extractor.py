"""Image and diagram extraction using GPT-4o vision."""

import base64
import hashlib
import os
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _render_page_as_image(page: fitz.Page, dpi: int = 150) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _encode_image_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _describe_image_with_gpt4o(image_b64: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a lecture slide. Describe all diagrams, "
                            "charts, figures, and equations in detail. "
                            "If no visuals exist, respond with SKIP."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


def extract_images_from_pdf(
    pdf_path: str,
    max_pages: int = 20,
) -> list[dict]:
    """Extract visual content from PDF pages using GPT-4o vision."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pdf_path = Path(pdf_path)

    results = []
    doc = fitz.open(str(pdf_path))
    pages_to_process = min(len(doc), max_pages)

    for page_num in range(pages_to_process):
        page = doc[page_num]
        print(f"  Page {page_num + 1}/{pages_to_process}...", end=" ")

        image_bytes = _render_page_as_image(page)
        image_b64 = _encode_image_base64(image_bytes)
        description = _describe_image_with_gpt4o(image_b64, client)

        if description.strip().upper() == "SKIP":
            print("no visuals, skipped")
            continue

        chunk_hash = hashlib.md5(
            f"{pdf_path.name}_img_{page_num + 1}".encode()
        ).hexdigest()

        results.append(
            {
                "text": f"[Visual content on page {page_num + 1}]: {description}",
                "page_num": page_num + 1,
                "source_file": pdf_path.name,
                "content_type": "image_description",
                "chunk_hash": chunk_hash,
            }
        )
        print("extracted")

    doc.close()
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_extractor.py path/to/file.pdf")
        sys.exit(1)

    results = extract_images_from_pdf(sys.argv[1], max_pages=3)
    print(f"\nExtracted {len(results)} pages with visual content")
    for r in results:
        print(f"\n--- Page {r['page_num']} ---")
        print(r["text"][:400])
