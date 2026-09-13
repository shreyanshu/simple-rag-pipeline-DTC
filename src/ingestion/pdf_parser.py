from pathlib import Path
import fitz


def parse_pdf(pdf_path: str) -> list[dict]:
    """Extract selectable text page-by-page from any PDF."""
    path = Path(pdf_path)
    pages = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "page": page_number,
                    "source": path.name,
                })

    return pages
