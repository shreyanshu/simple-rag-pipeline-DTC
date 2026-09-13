from src.ingestion.chunker import create_chunks


def test_fixed_chunking():
    pages = [{"text": "a" * 2500, "page": 1, "source": "x.pdf"}]
    chunks = create_chunks(pages, "fixed", chunk_size=1000, overlap=200)
    assert len(chunks) == 3


def test_page_chunking():
    pages = [
        {"text": "hello", "page": 1, "source": "x.pdf"},
        {"text": "world", "page": 2, "source": "x.pdf"},
    ]
    chunks = create_chunks(pages, "page")
    assert len(chunks) == 2
