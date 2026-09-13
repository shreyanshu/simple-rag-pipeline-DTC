from pinecone import Pinecone, ServerlessSpec
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE


def get_index(dimension: int):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [x["name"] if isinstance(x, dict) else x.name for x in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def upsert_chunks(index, chunks, embeddings, namespace=PINECONE_NAMESPACE):
    records = []
    for chunk, vector in zip(chunks, embeddings):
        records.append({
            "id": chunk.id,
            "values": vector,
            "metadata": {
                "text": chunk.text,
                "page": chunk.page,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            },
        })

    # Pinecone accepts batches; 100 is a convenient demo batch size.
    for start in range(0, len(records), 100):
        index.upsert(
            vectors=records[start:start + 100],
            namespace=namespace,
        )

    return len(records)
