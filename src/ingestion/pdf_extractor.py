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
    pdf_path = Path(pdf_path)  # the input when the function is called is the source/path to the lecture notes ex. data/raw/lecture.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")  # checks if the file exists

    pages = []  # list to store the page details (the dictionaries)
    doc = fitz.open(str(pdf_path))  # the file is opened and saved in this variable as a document

    for page_num, page in enumerate(doc, start=1):  # flips through each page in the document
        text = page.get_text().strip()  # extracts all the text and remove all the extra spaces at the beginning and end of the text

        if not text:
            continue  # if there is no text on the page, it skips to the next page

        chunk_hash = hashlib.md5(
            f"{pdf_path.name}_{page_num}_{text[:100]}".encode()
        ).hexdigest()  # creates a unique hash for this page using the file name, page # & the first 100 chars of text
        # (to avoid creating different hashes for the same page if the text changes slightly)

        pages.append(
            {
                "text": text,
                "page_num": page_num,
                "source_file": pdf_path.name,
                "content_type": "text",
                "chunk_hash": chunk_hash,
            }
        )  # append a dictionary with page details to the list of pages

    doc.close()  # close the document once all the pages are iterated through and saved in the page list
    return pages  # returns the list of page dictionaries with the extracted text and metadata


# we have the main function here to allow us to run this module directly for testing purposes.
# It takes a PDF file path as a command line argument, extracts the text, and prints out the number of
# pages extracted along with a preview of the first 2 pages.
# you can run it from the command line like this: python src/ingestion/pdf_extractor.py data/raw/lecture.pdf
if __name__ == "__main__":
    import sys

    results = extract_text_from_pdf(sys.argv[1])
    print(f"Extracted {len(results)} pages")
    for r in results[:2]:
        print(f"\n--- Page {r['page_num']} ---")
        print(r["text"][:300])
