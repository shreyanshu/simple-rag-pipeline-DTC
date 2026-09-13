# Simple PDF RAG

A deliberately transparent RAG pipeline for arbitrary text-based PDFs.

## Architecture

PDF -> PyMuPDF -> Chunking -> `all-MiniLM-L6-v2` -> Pinecone
                                                        |
Question -> `all-MiniLM-L6-v2` -> Pinecone top-K -------+
                                                        |
                                                        v
                                                   Gemini

Embeddings run locally using Hugging Face / Sentence Transformers.
Gemini is used only for answer generation.

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
```

Set `GOOGLE_API_KEY` and `PINECONE_API_KEY` in `.env`.

## Ingest

```bash
python scripts/ingest.py data/document.pdf --strategy recursive
```

Strategies:

```text
fixed      character-based chunks with overlap
recursive  paragraphs -> lines -> sentences -> words
page       one chunk per page
```

## Ask

```bash
python scripts/ask.py "What is the main topic of the document?"
```

## Streamlit

```bash
streamlit run app.py
```

## Notes

- `all-MiniLM-L6-v2` produces 384-dimensional embeddings.
- The Pinecone index is created automatically if it does not exist.
- The sample uses AWS `us-east-1` for the Pinecone serverless index; change
  `get_index()` if you need another supported region.
- The parser extracts selectable PDF text. Scanned/image-only PDFs need OCR,
  which is intentionally outside this basic project.
- For production, add document IDs/namespaces, deletion/update jobs, batching,
  retries, metadata filters, and evaluation.
