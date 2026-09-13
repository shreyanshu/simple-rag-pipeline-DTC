import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    source: str
    chunk_index: int


def fixed_chunks(pages, chunk_size=1000, overlap=200):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    counter = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(
                    id=f"{page['source']}-p{page['page']}-c{counter}",
                    text=piece,
                    page=page["page"],
                    source=page["source"],
                    chunk_index=counter,
                ))
                counter += 1
            if end == len(text):
                break
            start += chunk_size - overlap
    return chunks


def recursive_chunks(pages, chunk_size=1000, overlap=200):
    """Simple recursive splitter without a framework dependency."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    output = []

    def split_text(text, separator_index=0):
        if len(text) <= chunk_size:
            return [text]

        sep = separators[separator_index]
        parts = text.split(sep) if sep else list(text)
        pieces, current = [], ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                if len(part) > chunk_size and separator_index < len(separators) - 1:
                    pieces.extend(split_text(part, separator_index + 1))
                    current = ""
                else:
                    current = part

        if current:
            pieces.append(current)
        return pieces

    counter = 0
    for page in pages:
        pieces = split_text(page["text"])
        # lightweight character overlap between neighboring chunks
        merged = []
        for i, piece in enumerate(pieces):
            if i and overlap:
                prev = pieces[i - 1]
                piece = prev[-overlap:] + " " + piece
            merged.append(piece.strip())

        for piece in merged:
            if piece:
                output.append(Chunk(
                    id=f"{page['source']}-p{page['page']}-c{counter}",
                    text=piece,
                    page=page["page"],
                    source=page["source"],
                    chunk_index=counter,
                ))
                counter += 1
    return output


def page_chunks(pages):
    return [
        Chunk(
            id=f"{p['source']}-p{p['page']}-c0",
            text=p["text"],
            page=p["page"],
            source=p["source"],
            chunk_index=0,
        )
        for p in pages
    ]


def create_chunks(pages, strategy="recursive", chunk_size=1000, overlap=200):
    if strategy == "fixed":
        return fixed_chunks(pages, chunk_size, overlap)
    if strategy == "recursive":
        return recursive_chunks(pages, chunk_size, overlap)
    if strategy == "page":
        return page_chunks(pages)
    raise ValueError("strategy must be one of: fixed, recursive, page")
