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
    """Render a PDF page as a PNG image in memory."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _encode_image_base64(image_bytes: bytes) -> str:
    """Convert PNG bytes to a base64 string for the API request."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _needs_vision(page: fitz.Page) -> bool:
    """True if the page likely holds content that plain text extraction misses."""
    text = page.get_text().strip()
    has_images = len(page.get_images(full=True)) > 0
    has_drawings = len(page.get_drawings()) > 20
    return has_images or has_drawings or len(text) < 200


def _describe_image_with_gpt4o(image_b64: str, client: OpenAI) -> str:
    """Ask GPT-4o to transcribe and describe any visuals on the page."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a lecture slide. Transcribe every text label in any "
                            "diagram, chart, map or figure, and describe how the items are "
                            "arranged (what is at the center, which items are grouped "
                            "together, what the axes are, which items sit at the top, "
                            "bottom, left and right). If no visuals exist, respond with SKIP."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content


def extract_images_from_pdf(
    pdf_path: str,
    max_pages: int = 60,  # safety cap on cost
) -> list[dict]:
    """Extract visual content from PDF pages using GPT-4o vision."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pdf_path = Path(pdf_path)

    results = []
    doc = fitz.open(str(pdf_path))

    candidates = [i for i in range(len(doc)) if _needs_vision(doc[i])][:max_pages]
    print(f"  {len(candidates)} of {len(doc)} pages need vision")

    for page_num in candidates:
        page = doc[page_num]
        print(f"  Page {page_num + 1}...", end=" ")

        image_bytes = _render_page_as_image(page)
        image_b64 = _encode_image_base64(image_bytes)
        description = _describe_image_with_gpt4o(image_b64, client)

        if description.strip().upper() == "SKIP":
            print("no visuals, skipped")
            continue

        chunk_hash = hashlib.md5(f"{pdf_path.name}_img_{page_num + 1}".encode()).hexdigest()

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