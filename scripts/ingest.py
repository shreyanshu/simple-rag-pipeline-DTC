import argparse
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import create_chunks
from src.ingestion.embedder import LocalEmbedder
from src.ingestion.pinecone_injector import get_index, upsert_chunks


def main():
    parser = argparse.ArgumentParser(description="Inject a PDF into Pinecone.")
    parser.add_argument("pdf", help="Path to PDF")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "page"], default="recursive")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    pages = parse_pdf(args.pdf)
    print(f"Pages with text: {len(pages)}")

    chunks = create_chunks(
        pages,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Chunks: {len(chunks)}")

    embedder = LocalEmbedder()
    print(f"Embedding model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.dimension}")

    embeddings = embedder.embed_documents([c.text for c in chunks])

    index = get_index(embedder.dimension)
    count = upsert_chunks(index, chunks, embeddings)

    print(f"Indexed {count} chunks into Pinecone.")


if __name__ == "__main__":
    main()
