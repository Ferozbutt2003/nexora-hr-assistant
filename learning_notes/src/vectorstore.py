"""
vectorstore.py - Steps 4 & 5: embeddings (text -> numbers) and the Pinecone vector store.
"""
# Docstring: describes the file.

from pathlib import Path
# Path helps us take just the file name from a full path.

from pinecone import Pinecone
# The official Pinecone client: used here for admin tasks like clearing the index.

from langchain_huggingface import HuggingFaceEmbeddings
# Our local embedding model wrapper (Step 4).

from langchain_pinecone import PineconeVectorStore
# LangChain's bridge to Pinecone: handles embedding + uploading + searching for us.

from src import config
# Our settings file.


def get_embeddings():
    """Return the embedding model that turns text into 384 numbers."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,             # all-MiniLM-L6-v2
        encode_kwargs={"normalize_embeddings": True},  # same length vectors -> clean cosine scores
    )


def get_vectorstore():
    # Returns a "connection" to our Pinecone index that knows how to embed text.

    """Connect to the Pinecone index, using our embedding model."""
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,   # which index: "rag-course"
        embedding=get_embeddings(),              # model used to embed chunks AND questions
        pinecone_api_key=config.PINECONE_API_KEY,  # our key from .env (via config.py)
    )
    # IMPORTANT: chunks and questions MUST use the SAME embedding model,
    # otherwise they live on different "maps" and similarity means nothing.


def clear_index():
    # Deletes every record in the index, so re-indexing starts fresh (no ghosts).

    """Remove all old vectors from the index."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    # Connect to our Pinecone account.

    index = pc.Index(config.PINECONE_INDEX_NAME)
    # Get a handle to our specific index.

    try:
        # "try" = attempt this; if it fails, jump to "except" instead of crashing.
        index.delete(delete_all=True)
        # Delete every record in the index.
        print("Cleared old vectors from Pinecone.")
    except Exception:
        # Pinecone raises an error if the index is already empty (first run). That's fine.
        print("Index was already empty, nothing to clear.")


def index_chunks(chunks):
    # Takes our chunk Documents, cleans them, and uploads them to Pinecone.

    """Clean metadata, create fixed IDs, and upload chunks to Pinecone."""
    for c in chunks:
        # Loop through each chunk...
        c.metadata = {
            # ...and REPLACE its metadata with only the fields we need:
            "source": Path(c.metadata["source"]).name,  # "data\\Nexora...pdf" -> "Nexora...pdf"
            "page": c.metadata["page_label"],           # human page number ("2", "3"...)
            "chunk_id": c.metadata["chunk_id"],         # our chunk counter from Step 3
        }

    ids = [f"{c.metadata['source']}-{c.metadata['chunk_id']}" for c in chunks]
    # Build a fixed ID for every chunk, e.g. "Nexora_Employee_Handbook.pdf-4".
    # Same chunk = same ID every run, so uploads overwrite instead of duplicating.

    vectorstore = get_vectorstore()
    # Connect to Pinecone.

    vectorstore.add_documents(chunks, ids=ids)
    # For every chunk: embed its text -> upload (ID + vector + metadata + text). One line!

    return len(ids)
    # Return how many chunks we uploaded, so ingest.py can report it.