import tempfile
import streamlit as st

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import create_chunks
from src.ingestion.embedder import LocalEmbedder
from src.ingestion.pinecone_injector import get_index, upsert_chunks
from src.retrieval.retriever import Retriever
from src.generation.gemini import GeminiGenerator


st.set_page_config(page_title="Simple PDF RAG", page_icon="📚")
st.title("📚 Simple PDF RAG")
st.caption("Local all-MiniLM-L6-v2 embeddings + Pinecone + Gemini")

@st.cache_resource
def load_embedder():
    return LocalEmbedder()

embedder = load_embedder()

with st.sidebar:
    strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "page"])
    chunk_size = st.number_input("Chunk size", 200, 5000, 1000, 100)
    overlap = st.number_input("Overlap", 0, 1000, 200, 50)
    top_k = st.slider("Top-K", 1, 10, 5)

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])

if uploaded and st.button("Index PDF"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        pdf_path = tmp.name

    pages = parse_pdf(pdf_path)
    chunks = create_chunks(pages, strategy, chunk_size, overlap)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    index = get_index(embedder.dimension)
    count = upsert_chunks(index, chunks, embeddings)

    st.success(f"Indexed {count} chunks from {len(pages)} pages.")

question = st.chat_input("Ask a question about the indexed PDF")

if question:
    index = get_index(embedder.dimension)
    retriever = Retriever(index, embedder)
    matches = retriever.search(question, top_k)

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        answer = GeminiGenerator().answer(question, matches)
        st.write(answer)

        with st.expander("Retrieved chunks"):
            for i, match in enumerate(matches, 1):
                md = match.metadata or {}
                st.write(
                    f"**{i}. Page {md.get('page')} — score {match.score:.4f}**"
                )
                st.write(md.get("text", ""))
