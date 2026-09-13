from src.config import PINECONE_NAMESPACE


class Retriever:
    def __init__(self, index, embedder, namespace=PINECONE_NAMESPACE):
        self.index = index
        self.embedder = embedder
        self.namespace = namespace

    def search(self, question, top_k=5):
        vector = self.embedder.embed_query(question)
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        return result.matches
