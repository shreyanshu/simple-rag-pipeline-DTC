from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


class LocalEmbedder:
    """Local Hugging Face/Sentence Transformers embedding model."""

    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
