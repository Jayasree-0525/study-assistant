"""Image and diagram extraction using GPT-4o vision."""

import base64
import hashlib
import os
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # this reads the .env file and loads the OPENAI_API_KEY into the environment


# this function takes a pdf page and converts it into a png image in memory
def _render_page_as_image(page: fitz.Page, dpi: int = 150) -> bytes:
    mat = fitz.Matrix(
        dpi / 72, dpi / 72
    )  # DPI = Dots Per Inch. PDF page is natively 72 DPI. 150 DPI = rendering it at roughly 2x sharpness -> clear enough for GPT-4o to read diagrams and text
    pix = page.get_pixmap(matrix=mat)  # renders the page as pixels
    return pix.tobytes("png")


# this function takes the raw png image bytes and converts it into a base64 string so it can be sent in a JSON api request
def _encode_image_base64(image_bytes: bytes) -> str:
    # PNG bytes are just an image stored as raw binary data
    return base64.b64encode(image_bytes).decode("utf-8")


# this function sends the base64 encoded image to GPT-4o with a prompt asking it to describe any visuals on the page.
# It then gets a text description of any visuals on the page
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
                            "detail": "low",  # GPT-4o looks at a compressed 512x512 version of the image. Fast and cheap.
                        },
                    },
                ],
            }
        ],
        max_tokens=500,  # caps the response, so GPT-4o can't write an essay about a single slide
    )
    return response.choices[0].message.content


# this is the main function that gets called and calls the other functions above.
def extract_images_from_pdf(
    # everything comes togther here.
    pdf_path: str,
    max_pages: int = 20,  # strategy to control cost
) -> list[dict]:
    """Extract visual content from PDF pages using GPT-4o vision."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pdf_path = Path(pdf_path)

    results = []
    doc = fitz.open(str(pdf_path))
    pages_to_process = min(len(doc), max_pages)

    # for every page in the document it renders the pages as an image using helper function 1,
    # then encodes to Base64 using function 2 & sends to GPT-4o to get a description
    for page_num in range(pages_to_process):
        page = doc[page_num]
        print(f"  Page {page_num + 1}/{pages_to_process}...", end=" ")

        image_bytes = _render_page_as_image(page)  # helper function 1
        image_b64 = _encode_image_base64(image_bytes)  # helper function 2
        description = _describe_image_with_gpt4o(image_b64, client)  # helper function 3

        # if GPT-4o says no visuals skip, then continue. otherwise build a dictionary & append results
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


# lets us run the file from the terminal for testing
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
