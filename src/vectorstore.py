"""Embedding model and Pinecone vector store helpers."""
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src import config


def get_embeddings():
    """Return the local embedding model (normalized vectors for cosine similarity)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore():
    """Connect to the Pinecone index. Chunks and questions must use the same embeddings."""
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=config.PINECONE_API_KEY,
    )


def clear_index():
    """Delete all vectors so re-indexing never leaves outdated chunks behind."""
    index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True)
        print("Cleared old vectors from Pinecone.")
    except Exception:
        # Pinecone raises an error when the index is already empty.
        print("Index was already empty, nothing to clear.")


def index_chunks(chunks):
    """Upload chunks with clean metadata and stable IDs. Returns the number uploaded."""
    for c in chunks:
        c.metadata = {
            "source": Path(c.metadata["source"]).name,
            "page": c.metadata["page_label"],
            "chunk_id": c.metadata["chunk_id"],
        }

    # Stable IDs (file name + chunk number) make re-uploads overwrite instead of duplicate.
    ids = [f"{c.metadata['source']}-{c.metadata['chunk_id']}" for c in chunks]
    get_vectorstore().add_documents(chunks, ids=ids)
    return len(ids)
