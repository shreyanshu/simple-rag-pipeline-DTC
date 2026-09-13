import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-rag-demo")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Copy .env.example to .env.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Copy .env.example to .env.")
