import argparse
from src.ingestion.embedder import LocalEmbedder
from src.ingestion.pinecone_injector import get_index
from src.retrieval.retriever import Retriever
from src.generation.gemini import GeminiGenerator


def main():
    parser = argparse.ArgumentParser(description="Ask a question about indexed PDFs.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embedder = LocalEmbedder()
    index = get_index(embedder.dimension)
    retriever = Retriever(index, embedder)

    matches = retriever.search(args.question, args.top_k)

    print("\nRETRIEVED CHUNKS")
    print("=" * 60)
    for i, match in enumerate(matches, 1):
        md = match.metadata or {}
        print(f"{i}. score={match.score:.4f} | page={md.get('page')} | source={md.get('source')}")
        print(md.get("text", "")[:500].replace("\n", " "))
        print()

    generator = GeminiGenerator()
    answer = generator.answer(args.question, matches)

    print("ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
