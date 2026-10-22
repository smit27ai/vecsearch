"""Read PDFs into pages, and split pages into overlapping chunks.

Chunks are counted in words, not characters, so a chunk never ends in the
middle of a word. Neighbouring chunks overlap, so a sentence cut at a chunk
border still appears whole in one of the two chunks.
"""

import re
from pathlib import Path


def extract_pages(path):
    """[(page_number starting at 1, text)] for every page that has text."""
    from pypdf import PdfReader

    pages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = clean(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


def clean(text):
    text = re.sub(r"-\n(\w)", r"\1", text)  # join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(pages, source, size=120, overlap=30):
    """Split each page into chunks of `size` words, `overlap` words shared.

    Returns [{"id", "text", "source", "page", "chunk"}]. Chunks stay inside
    one page, so every chunk can cite the page it came from.
    """
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    stem = Path(source).stem
    chunks = []
    for page, text in pages:
        words = text.split()
        step = size - overlap
        for n, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
            piece = words[start:start + size]
            chunks.append({
                "id": f"{stem}_p{page}_c{n}",
                "text": " ".join(piece),
                "source": Path(source).name,
                "page": page,
                "chunk": n,
            })
    return chunks
